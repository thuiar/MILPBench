import argparse
import pickle
from pathlib import Path
from typing import Union
import os
import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_metric_learning import losses

from model.graphcnn import GNNPolicy

__all__ = ["train"]

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
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment = assignment

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
            [variable_features, constraint_features, edge_indices, edge_features, solution] = pickle.load(f)

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.FloatTensor(solution)
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = len(constraint_features) + len(variable_features)
        graph.cons_nodes = len(constraint_features)
        graph.vars_nodes = len(variable_features)

        return graph


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output

def process(policy, data_loader, device, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            #print("QwQ")
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits, select = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            
            n = len(batch.variable_features)
            choose = {}
            for i in range(n):
                if(select[i] >= 0.5):
                    choose[i] = 0
                else:
                    choose[i] = 1
            new_idx_train = []
            for i in range(n):
                if(choose[i]):
                    new_idx_train.append(i)
            
            set_c = 0.7
            if(len(new_idx_train) < set_c * n):
                loss_select = (set_c - len(new_idx_train) / n) ** 2
            else:
                loss_select = 0
            #print(batch.constraint_features)
            #print(batch.edge_index)
            #print(batch.edge_attr)
            #print(batch.variable_features)
            # Index the results by the candidates, and split and pad them
            # logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            #loss = F.binary_cross_entropy(logits, batch.assignment)
            loss_func = torch.nn.MSELoss()
            #print(logits)
            #print(logits)
            #print(batch.assignment)
            #print(logits)
            loss = loss_func(logits[new_idx_train], batch.assignment[new_idx_train]) + loss_select
            
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            mean_loss += loss.item() * batch.num_graphs
            # mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    # mean_acc /= n_samples_processed
    return mean_loss

def train(
    path: str,
    model_save_path: Union[str, Path],
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    This function trains a GNN policy on training data. 

    Args:
        data_path: Path to the data directory.
        model_save_path: Path to save the model.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of epochs to train for.
        device: Device to use for training.
    """
    #训练路径
    train_data_path = f'instances/{path}/train'
    # load samples from data_path and divide them
    sample_files = [str(path) for path in Path(train_data_path).glob("pair*.pickle")]
    print(sample_files)
    train_files = sample_files[: int(0.9 * len(sample_files))]
    valid_files = sample_files[int(0.9 * len(sample_files)) :]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=batch_size, shuffle = False)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.loader.DataLoader(valid_data, batch_size=batch_size, shuffle = False)

    policy = GNNPolicy().to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loss = process(policy, train_loader, device, optimizer)
        #valid_loss = process(policy, valid_loader, device, None)
        valid_loss = 0
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:0.3f}, Valid Loss: {valid_loss:0.3f}")
    model_save_path = f'{model_save_path}/{path}_trained.pkl'
    torch.save(policy.state_dict(), model_save_path)
    print(f"Trained parameters saved to {model_save_path}")

def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="fc.data", help="Path for train Data.")
    parser.add_argument("--model_save_path", type=str, default="trained_model", help="Path to save the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train for.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
