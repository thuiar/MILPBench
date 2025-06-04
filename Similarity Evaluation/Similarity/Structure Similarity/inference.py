import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from model import SubGAE
from utils import create_optimizer, build_args, load_configs, set_seed
import networkx as nx
from torch_geometric.utils import to_networkx
import community.community_louvain as community_louvain
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
import os
import pickle
import random

def main(args):
    set_seed(args.seed)
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
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

    dataset_path = args.input_dir
    datalist = os.listdir(dataset_path)
    namelist = []
    z = torch.zeros((len(datalist), args.enc_out_channels))
    cnt = 0

    for data in datalist:
        namelist.append(data)
        with open(os.path.join(dataset_path, data), 'rb') as f:
            pkl = pickle.load(f)
        
        data = pkl.to(device)
        with torch.no_grad():
            origin_z = model.encoder(data.x, data.edge_index, data.edge_attr)
            z[cnt] = torch.mean(origin_z, dim=0)
        
        cnt += 1
        # print(f"cnt={cnt}")
        torch.cuda.empty_cache()

    with open(args.output_file, "wb") as f:
        pickle.dump(z, f)
    
    with open(args.filename, "wb") as f:
        pickle.dump(namelist, f)

if __name__ == "__main__":
    args = build_args()
    args = load_configs(args, args.cfg_path)
    print(args)
    main(args)
