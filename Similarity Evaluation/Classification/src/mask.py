import torch.nn
import numpy as np

class GraphMask(torch.nn.Module):
    # TODO
    def __init__(self, p_node, p_edge, feat_dim):
        super(GraphMask, self).__init__()
        self.p_node = p_node
        self.p_edge = p_edge
        # self.mask_token = torch.nn.Parameter(torch.zeros(1, feat_dim))
        self.mask_token = torch.ones(1, feat_dim)

    def mask_node(self, subgraph, feat):
        S = torch.tensor(list(subgraph), device=feat.device)
        random_numbers = torch.rand(S.shape[0], device=feat.device)
        masked_nodes = S[random_numbers <= self.p_node]
        # x = feat.clone()
        # x[masked_nodes] = self.mask_token.to(feat.device)
        # return x, masked_nodes
        feat[masked_nodes] = self.mask_token.to(feat.device)
        return feat, masked_nodes

    def drop_edge(self, subgraph, edge_idx, edge_attr, mode):
        S = torch.tensor(list(subgraph), device=edge_idx.device)
        mask_subgraph = torch.isin(edge_idx, S).any(dim=0)
        if mode == 1:
            mask_random = torch.rand(mask_subgraph.size(), device=edge_idx.device) <= self.p_edge
            edge_mask = mask_subgraph & mask_random
        else:
            edge_mask = mask_subgraph
        return edge_idx[:, ~edge_mask], edge_idx[:, edge_mask], edge_attr[~edge_mask], edge_attr[edge_mask]
    
class FeatMask(torch.nn.Module):
    # TODO
    def __init__(self):
        super(FeatMask, self).__init__()

    def forward(self, x):
        return x