import networkx as nx
import community 
import math
import os
import argparse, pickle
from torch_geometric.data import Data
from torch_geometric.utils import degree
import statistics
import numpy as np
import torch
from scipy.spatial import distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy


def normalize(data):
    total = sum(data)
    return [x / total for x in data]

# order
# 0.coef_dens
# 1.cons_degree_mean
# 2.cons_degree_std
# 3.var_degree_mean
# 4.var_degree_std
# 5.lhs_mean
# 6.lhs_std
# 7.rhs_mean
# 8.rhs_std
# 9.clustering_coef
# 10.modularity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    eps = 0.000001
    parser.add_argument('--type', choices=["slack", "direct", "sta"])
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    

    A = [[] for i in range(10)]
    for file in os.listdir(args.input_dir):
        with open(os.path.join(args.input_dir, file), 'rb') as f:
            data = pickle.load(f)
        
        A[0].append(data.edge_index.shape[1]/2 / data.num_vars / data.num_constrs)
        deg = degree(index=data.edge_index[1].flatten(), num_nodes=data.num_vars+data.num_constrs)
        var_deg = deg[:data.num_vars].tolist()
        constr_deg = deg[data.num_vars:].tolist()
        A[1].append(statistics.mean(constr_deg))
        A[2].append(statistics.stdev(constr_deg))
        A[3].append(statistics.mean(var_deg))
        A[4].append(statistics.stdev(var_deg))
        lhs_weights = data.edge_attr[0::2].tolist()
        A[5].append(np.mean(lhs_weights))
        A[6].append(np.std(lhs_weights))
        con_edges = {}
        rhs_weights = []
        for i in range(data.edge_index.size(1)):
            if i % 2 == 1:
                continue
            con_idx = data.edge_index[1, i].item() - data.num_vars
            var_idx = data.edge_index[0, i].item()
            coef = data.edge_attr[i].item()
            if con_idx >= 0 and abs(coef) > eps:
                if con_idx not in con_edges:
                    con_edges[con_idx] = []
                con_edges[con_idx].append((var_idx, coef))
        var_nodes = data.x[:data.num_vars]
        con_nodes = data.x[data.num_vars:]
        for i, node in enumerate(con_nodes):
            con_feat = node.tolist()
            lhs_value = 0
            flag = 0
            if i in con_edges:
                for var_idx, coef in con_edges[i]:
                    flag = 1
            if con_feat[12] > 0.5:
                con_feat[10] = math.floor(con_feat[10] + 0.5)
            rhs_value = con_feat[10] + lhs_value
            if flag == 1:
                rhs_weights.append(rhs_value)
        A[7].append(np.mean(rhs_weights))
        A[8].append(np.std(rhs_weights))
        G = nx.Graph()
        for i in range(data.num_vars+data.num_constrs):
            G.add_node(i)
        for i in range(data.edge_index.size(1)):
            if i % 2 == 1:
                continue
            G.add_edge(data.edge_index[0, i].item(), data.edge_index[1, i].item())
        best_partition = community.best_partition(G)
        modularity = community.modularity(best_partition, G)
        A[9].append(modularity)
    
    print(f"Found {len(A[0])} entries in input_dir.")
    B = torch.tensor(A)
    B = B.transpose(0, 1)
    f = open(args.output_file, "wb")
    pickle.dump(B, f)
    f.close()