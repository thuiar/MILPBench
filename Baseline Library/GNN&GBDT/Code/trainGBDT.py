import numpy as np
import argparse
import pickle
import random
import os
from model.gbdt_regressor import GradientBoostingRegressor
import argparse
import pickle
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_metric_learning import losses

from model.graphcnn import GNNPolicy

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        assignment1,
        assignment2
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment1 = assignment1
        self.assignment2 = assignment2

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with open(self.sample_files[index], "rb") as f:
            [variable_features, constraint_features, edge_indices, edge_features, solution1, solution2] = pickle.load(f)
        solution2 = [int(x) for x in solution2]

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.LongTensor(solution1),
            torch.LongTensor(solution2),
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = len(constraint_features) + len(variable_features)
        graph.cons_nodes = len(constraint_features)
        graph.vars_nodes = len(variable_features)

        return graph


def train_GBDT(number:int, problem:str, difficulty:str,device:torch.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")):
    print("Tesing the performance of GBDT regressor...")
    # Load data
    model_path = './trained_model_' + problem + '.pkl'
    print("QwQ", model_path)
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    data = []
    label = []
    max_num = 30000
    now_num = 0
    for num in range(number):
        #查询data.pickle是否存在，若存在则读入
        if(os.path.exists('./example_' + problem + '/pair' + str(num) + '.pickle') == False):
            print("No problem file!", './example_' + problem + '/pair' + str(num) + '.pickle')
            return 
        with open('./example_' + problem + '/pair' + str(num) + '.pickle', "rb") as f:
            pair = pickle.load(f)
        
        File = []
        File.append('./example_' + problem + '/pair' + str(num) + '.pickle')
        file_data = GraphDataset(File)
        loader = torch_geometric.loader.DataLoader(file_data, batch_size = 1)
        for batch in loader:
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )

        now_data = logits.tolist()
        now_label = pair[5]
        p = max_num / (len(logits.tolist()) * number)
        for i in range(len(now_data)):
            if(random.random() <= p):
                data.append(now_data[i])
                label.append(now_label[i])
                now_num += 1
    # Train model
    print(now_num)
    reg = GradientBoostingRegressor()
    #print(data)
    reg.fit(data=np.array(data), label=np.array(label), n_estimators=30, learning_rate=0.1, max_depth=5, min_samples_split=2)
    # Model evaluation
    with open('./GBDT_' + problem + '.pickle', 'wb') as f:
        pickle.dump([reg], f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type = int, default = 10, help = 'The number of instances.')
    parser.add_argument("--problem", type = str, default = 'IS', help = 'The number of instances.')
    parser.add_argument("--difficulty", type = str, default = 'easy', help = 'The number of instances.')
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    train_GBDT(**vars(args))