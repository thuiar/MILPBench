import torch.nn as nn
import torch
import argparse
import os
import numpy as np
import random
from functools import partial
from texttable import Texttable
from torch import optim as optim
import yaml
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)
import logging
from tensorboardX import SummaryWriter
from loss import *


# TODO
# class Logger():


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

def load_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k or "scale" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use configs------")
    return args


def build_args():
    parser = argparse.ArgumentParser(description="SubGAE")

    # meta args
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--type", choices=["slack", "direct"])
    # TODO: modify later
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_path", type=str, default="./dataset/MILP_nursesched/data_raw")

    # SubGAE settings
    parser.add_argument("--bipartite", type=int, default=0)
    parser.add_argument("--has_edge_feat", type=int, default=0)
    parser.add_argument("--subgraph_limit", type=int, default=0)
    parser.add_argument("--node_mask_rate", type=float, default=0.5)
    parser.add_argument("--edge_mask_rate", type=float, default=0.5)
    parser.add_argument("--w_feat", type=float, default=0.5)
    parser.add_argument("--w_deg", type=float, default=0.5)
    parser.add_argument("--w_typ", type=float, default=0.5)
    parser.add_argument("--w_logit", type=float, default=0.5)
    parser.add_argument("--w_weight", type=float, default=0.5)
    parser.add_argument("--w_moe", type=float, default=0.5)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--act_expert_num", type=int, default=2)

    # learning settings
    parser.add_argument("--batch_size", type=int, default=2 ** 14)
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--div_epoch", type=int)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--grad_norm", type=float, default=1.0)

    # model settings
    parser.add_argument("--enc_in_channels", type=int, default=1433)
    parser.add_argument("--enc_hid_channels", type=int, default=128)
    parser.add_argument("--enc_out_channels", type=int, default=32)
    parser.add_argument("--enc_layer_num", type=int, default=2)
    parser.add_argument("--enc_dropout", type=float, default=0.2)
    parser.add_argument("--enc_norm", type=str, default="batchnorm")
    parser.add_argument("--enc_gnn", type=str, default="gcn")
    parser.add_argument("--enc_act", type=str, default="relu")

    parser.add_argument("--dec_in_channels", type=int, default=32)
    parser.add_argument("--dec_hid_channels", type=int, default=128)
    parser.add_argument("--dec_out_channels", type=int, default=1434)
    parser.add_argument("--dec_layer_num", type=int, default=2)
    parser.add_argument("--dec_dropout", type=float, default=0.2)
    parser.add_argument("--dec_norm", type=str, default="batchnorm")
    parser.add_argument("--dec_gnn", type=str, default="gcn")
    parser.add_argument("--dec_act", type=str, default="relu")

    parser.add_argument("--mlp_hid_channels", type=int, default=128)
    parser.add_argument("--mlp_layer_num", type=int, default=2)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_act", type=str, default="relu")
    
    parser.add_argument("--feat_loss", type=str, default="sce")
    parser.add_argument("--deg_loss", type=str, default="sce")

    # reconstruction settings
    parser.add_argument("--r_input_dir", type=str)
    parser.add_argument("--r_output_dir", type=str)
    parser.add_argument("--r_count", type=int)
    parser.add_argument("--r_prefix", type=str, default="")
    parser.add_argument("--r_id", type=int, default=1)

    # extrapolation settings
    parser.add_argument("--e_input_dir", type=str)
    parser.add_argument("--e_output_dir", type=str)
    parser.add_argument("--e_scale", type=float)
    parser.add_argument("--e_count", type=int)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--filename", type=str)

    args = parser.parse_args()
    return args

def create_gnn(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(first_channels, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(first_channels, second_channels, heads=heads)
    else:
        raise NotImplementedError
    return layer

def create_loss(name):
    if name == "auc":
        loss_fn = auc_loss
    elif name == "hinge_auc":
        loss_fn = hinge_auc_loss
    elif name == "log_rank":
        loss_fn = log_rank_loss
    elif name == "ce":
        loss_fn = ce_loss
    elif name == "info_nce":
        loss_fn = info_nce_loss
    elif name == "sce":
        loss_fn = sce_loss
    elif name == "sig":
        loss_fn = sig_loss
    elif name == "bce":
        loss_fn = F.binary_cross_entropy
    elif name == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise NotImplementedError
    return loss_fn

def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError

    return optimizer

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError
    
def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity
    
class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
    
def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()

def negative_sampling(n, m, subgraph_nodes, device):
    node_set_tensor = subgraph_nodes.clone().detach().to(device)
    # node_set_tensor = torch.tensor(subgraph_nodes, dtype=torch.int64, device=device)

    endpoints1 = torch.randint(0, n, (m,), device=device)
    endpoints2 = node_set_tensor[torch.randint(0, len(node_set_tensor), (m,), device=device)]

    swap = torch.rand(m, device=device) < 0.5
    swapped_endpoints1 = torch.where(swap, endpoints2, endpoints1)
    swapped_endpoints2 = torch.where(swap, endpoints1, endpoints2)

    edges = torch.vstack((swapped_endpoints1, swapped_endpoints2))
    return edges

