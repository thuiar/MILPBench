import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from SubGAE.model import SubGAE
from SubGAE.utils import create_optimizer, build_args, load_configs, set_seed
import networkx as nx
from torch_geometric.utils import to_networkx
from dataset.MILP_utils import postprocess
import community.community_louvain as community_louvain
import torch.nn.functional as F
import numpy as np
import os
import pickle
import random

def main(args):
    set_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    # To-Do: Deal has_edge_feat==0
    model = SubGAE(
        bipartite=args.bipartite,
        has_edge_feat=args.has_edge_feat,
        node_mask_rate=args.node_mask_rate,
        edge_mask_rate=args.edge_mask_rate,
        enc_in_channels=args.enc_in_channels,
        enc_hid_channels=args.enc_hid_channels,
        enc_out_channels=args.enc_out_channels,
        enc_layer_num=args.enc_layer_num,
        enc_dropout=args.enc_dropout,
        enc_norm=args.enc_norm,
        enc_gnn=args.enc_gnn,
        enc_act=args.enc_act,
        dec_in_channels=args.dec_in_channels,
        dec_hid_channels=args.dec_hid_channels,
        dec_out_channels=args.dec_out_channels,
        dec_layer_num=args.dec_layer_num,
        dec_dropout=args.dec_dropout,
        dec_norm=args.dec_norm,
        dec_gnn=args.dec_gnn,
        dec_act=args.dec_act,
        expert_num=args.expert_num,
        act_expert_num=args.act_expert_num,
        mlp_hid_channels=args.mlp_hid_channels,
        mlp_layer_num=args.mlp_layer_num,
        mlp_dropout=args.mlp_dropout,
        mlp_act=args.mlp_act,
        feat_loss_func=args.feat_loss,
        device=device
    )
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.model_path))

    data_list = os.listdir(args.r_input_dir)

    for data_name in data_list:
        print(f"Processing data {data_name}...")
        file = open(os.path.join(args.r_input_dir, data_name),'rb')
        data = pickle.load(file)
        file.close()

        for key, item in data:
            if torch.is_tensor(item) and item.requires_grad:
                item.requires_grad = False

        G = to_networkx(data, to_undirected=True)
        partition = community_louvain.best_partition(G)
        partit = []
        for i in range(data.num_nodes):
            partit.append(partition[i])

        max_id = max(partit)
        subgraph_node = []
        subgraph_count = 0

        node_lists = [[] for i in range(max_id+1)]
        for i in range(data.num_nodes):
            node_lists[partit[i]].append(i)

        for i in range(max_id+1):
            if len(node_lists[i]) > args.subgraph_limit:
                subgraph_node.append(node_lists[i])
                subgraph_count += 1

        data = data.to(device)
        data.rnode = 0
        for i in range(args.r_count):
            data, new_node_list = model.reconstruction_step(data=data, subgraph_count=subgraph_count,subgraph_node=subgraph_node, device=device, subgraph_id=args.r_id)
            
            if args.dataset == "MILP":
                data = postprocess(data, device=device, new_node_list=new_node_list, type=args.type)
                print("Postprocess Done.")
             
        file = open(os.path.join(args.r_output_dir, args.r_prefix+data_name), 'wb')
        print(f"Totally reconstructed {data.rnode} nodes.")
        pickle.dump(data, file)
        del data
        del new_node_list
        torch.cuda.empty_cache()
        file.close()

if __name__ == "__main__":
    args = build_args()
    # TODO: modify after finalizing hyperparameters
    args = load_configs(args, args.cfg_path)
    # args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)