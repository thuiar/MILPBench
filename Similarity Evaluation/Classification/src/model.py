import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from loss import ce_loss, sce_loss
from mask import GraphMask, FeatMask
from utils import create_activation, create_gnn, create_norm, create_loss, negative_sampling
from modules import GNNModule, MLPModule, GatingModule, cv_squared, SparseDispatcher
from torch_geometric.utils import degree, add_self_loops
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch.nn as nn
import random
from tqdm import tqdm
import math

class SubGAE(torch.nn.Module):
    def __init__(
        self, bipartite, has_edge_feat, node_mask_rate, edge_mask_rate,
        enc_in_channels, enc_hid_channels, enc_out_channels, enc_layer_num, enc_dropout, enc_norm, enc_gnn, enc_act,
        dec_in_channels, dec_hid_channels, dec_out_channels, dec_layer_num, dec_dropout, dec_norm, dec_gnn, dec_act,
        expert_num, act_expert_num, mlp_hid_channels, mlp_layer_num, mlp_dropout, mlp_act,
        feat_loss_func, device
    ):
        super(SubGAE, self).__init__()
        self.bipartite = bipartite
        self.has_edge_feat = has_edge_feat
        self.expert_num = expert_num
        self.graphMasker = GraphMask(node_mask_rate, edge_mask_rate, feat_dim=enc_in_channels)
        self.featMasker = FeatMask()

        self.encoder = GNNModule(
            in_channels=enc_in_channels,
            hidden_channels=enc_hid_channels,
            out_channels=enc_out_channels,
            num_layers=enc_layer_num,
            dropout=enc_dropout,
            norm=enc_norm,
            gnn=enc_gnn,
            activation=enc_act
        )
        self.enc2dec = nn.Linear(enc_out_channels, dec_in_channels, bias=False)
        self.expert_num = expert_num
        self.act_expert_num = act_expert_num
        self.selector = GatingModule(
            expert_num=expert_num, 
            act_expert_num=act_expert_num, 
            input_dim=dec_in_channels
        )

        self.linkDecoders = []
        self.typDecoders = []
        self.weightDecoders = []
        self.degreeDecoders = []
        self.featDecoder = GNNModule(
            in_channels=dec_in_channels,
            hidden_channels=dec_hid_channels,
            out_channels=dec_out_channels,
            num_layers=dec_layer_num,
            dropout=dec_dropout,
            norm=dec_norm,
            gnn=dec_gnn,
            activation=dec_act
        )

        for i in range(expert_num):
            self.linkDecoders.append(MLPModule(
                in_channels=dec_in_channels,
                hidden_channels=mlp_hid_channels,
                out_channels=1,
                num_layers=mlp_layer_num,
                dropout=mlp_dropout,
                activation=mlp_act
            ))

            self.typDecoders.append(MLPModule(
                in_channels=dec_in_channels,
                hidden_channels=mlp_hid_channels,
                out_channels=1,
                num_layers=mlp_layer_num,
                dropout=mlp_dropout,
                activation=mlp_act
            ))

            self.weightDecoders.append(MLPModule(
                in_channels=dec_in_channels,
                hidden_channels=mlp_hid_channels,
                out_channels=1,
                num_layers=mlp_layer_num,
                dropout=mlp_dropout,
                activation=mlp_act
            ))

            self.degreeDecoders.append(MLPModule(
                in_channels=dec_in_channels,
                hidden_channels=mlp_hid_channels,
                out_channels=1,
                num_layers=mlp_layer_num,
                dropout=mlp_dropout,
                activation=mlp_act
            ))

            self.linkDecoders[i].to(device)
            self.typDecoders[i].to(device)
            self.weightDecoders[i].to(device)
            self.degreeDecoders[i].to(device)

        self.featDecoder.to(device)
        self.feat_criterion = create_loss(feat_loss_func)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.enc2dec.reset_parameters()
        self.selector.reset_parameters()
        self.featDecoder.reset_parameters()

        for i in range(self.expert_num):
            self.linkDecoders[i].reset_parameters()
            self.typDecoders[i].reset_parameters()
            self.weightDecoders[i].reset_parameters()
            self.degreeDecoders[i].reset_parameters()

    @torch.no_grad()
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)
        return z

    def train_step(
        self, data: Data, optimizer,
        w_feat, w_deg, w_typ, w_weight, w_logit, w_moe,
        batch_size=2 ** 16, grad_norm=1.0,
    ):
        # Train all nodes
        subgraph_nodes = torch.arange(data.num_nodes, device=data.x.device)
        # subgraph_id = random.randint(0, data.subgraph_count - 1)
        # subgraph_nodes = data.subgraph_node[subgraph_id]
        
        if self.has_edge_feat == 1:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr_max = torch.max(edge_attr)
            edge_attr_min = torch.min(edge_attr)
            if edge_attr_max - edge_attr_min < 0.0001:
                edge_attr_max += 1
                edge_attr_min -= 1
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, None
        masked_x, masked_nodes = self.graphMasker.mask_node(subgraph=subgraph_nodes, feat=x)
        remain_edge_index, mask_edge_index, remain_edge_attr, mask_edge_attr = self.graphMasker.drop_edge(subgraph=subgraph_nodes, edge_idx=edge_index, edge_attr=edge_attr, mode=1)        

        neg_edges = negative_sampling(
            n=data.num_nodes, 
            m=mask_edge_index.view(2, -1).size(1), 
            subgraph_nodes=masked_nodes,
            device=mask_edge_index.device
        ).view_as(mask_edge_index)

        total_loss = 0
        total_feat_loss = 0
        total_deg_loss = 0
        total_logit_loss = 0
        total_typ_loss = 0
        total_weight_loss = 0
        total_moe_loss = 0
        deg_ori = degree(remain_edge_index[1].flatten(), data.num_nodes).float()
        deg = deg_ori / torch.max(deg_ori)

        for perm in DataLoader(
            range(mask_edge_index.size(1)), batch_size=batch_size, shuffle=True
        ):
            optimizer.zero_grad()
            moe_loss = 0
            origin_z = self.encoder(masked_x, remain_edge_index, remain_edge_attr)
            projected_z = self.enc2dec(origin_z)
            masked_z = self.featMasker(projected_z)

            # feat reconstruction
            recon_x = self.featDecoder(masked_z, remain_edge_index, remain_edge_attr)
            # TODO: choose loss calculation way
            feat_loss = self.feat_criterion(recon_x[masked_nodes], x[masked_nodes])

            # degree reconstruction
            deg_gates, deg_load = self.selector(x=projected_z[subgraph_nodes], train=True)
            importance = deg_gates.sum(0)
            moe_loss += cv_squared(importance) + cv_squared(deg_load)
            deg_dispatcher = SparseDispatcher(self.expert_num, deg_gates)

            deg_inputs = deg_dispatcher.dispatch(projected_z[subgraph_nodes]) 
            deg_outputs = [self.degreeDecoders[i](x=deg_inputs[i], sigmoid=True) for i in range(self.expert_num)]

            # TODO: choose loss function
            deg_loss = F.mse_loss(deg[subgraph_nodes], torch.squeeze(deg_dispatcher.combine(deg_outputs), dim=1))

            # link reconstruction
            batch_pos_edge = mask_edge_index[:, perm]
            batch_neg_edge = neg_edges[:, perm]
            pos_input = projected_z[batch_pos_edge[0]] * projected_z[batch_pos_edge[1]]
            neg_input = projected_z[batch_neg_edge[0]] * projected_z[batch_neg_edge[1]]
            # pos_input = torch.cat((projected_z[batch_pos_edge[0]], projected_z[batch_pos_edge[1]]), dim=1)
            # neg_input = torch.cat((projected_z[batch_neg_edge[0]], projected_z[batch_neg_edge[1]]), dim=1)

            pos_gates, pos_load = self.selector(x=pos_input, train=True)
            pos_importance = pos_gates.sum(0)
            moe_loss += cv_squared(pos_importance) + cv_squared(pos_load)
            pos_dispatcher = SparseDispatcher(self.expert_num, pos_gates)
            pos_inputs = pos_dispatcher.dispatch(pos_input)
            pos_outputs = [self.linkDecoders[i](x=pos_inputs[i], sigmoid=True) for i in range(self.expert_num)]
            pos_output = pos_dispatcher.combine(pos_outputs)
            
            neg_gates, neg_load = self.selector(x=projected_z[batch_neg_edge[0]], train=True)
            neg_importance = neg_gates.sum(0)
            moe_loss += cv_squared(neg_importance) + cv_squared(neg_load)
            neg_dispatcher = SparseDispatcher(self.expert_num, neg_gates)
            neg_inputs = neg_dispatcher.dispatch(neg_input)
            neg_outputs = [self.linkDecoders[i](x=neg_inputs[i], sigmoid=True) for i in range(self.expert_num)]
            neg_output = neg_dispatcher.combine(neg_outputs)
            # TODO: Try other loss functions
            logit_loss = ce_loss(pos_out=pos_output, neg_out=neg_output)

            # typ reconstruction
            typ = ((mask_edge_attr[perm] - torch.floor(mask_edge_attr[perm])) == 0).float()
            typ_outputs = [self.typDecoders[i](x=pos_inputs[i], sigmoid=True) for i in range(self.expert_num)]
            typ_output = pos_dispatcher.combine(typ_outputs)
            # print(typ)
            # print(typ_output)
            typ_loss = F.mse_loss(typ, typ_output)

            # weight reconstruction
            weight = (mask_edge_attr[perm] - edge_attr_min) / (edge_attr_max - edge_attr_min)
            weight_outputs = [self.weightDecoders[i](x=pos_inputs[i], sigmoid=True) for i in range(self.expert_num)]
            weight_output = pos_dispatcher.combine(weight_outputs)
            # print(weight)
            # print(weight_output)
            # TODO: Try other loss functions
            weight_loss = F.mse_loss(weight, weight_output)

            loss = w_feat * feat_loss + w_deg * deg_loss + w_logit * logit_loss + w_typ * typ_loss + w_weight * weight_loss + w_moe * moe_loss
            loss.backward()
            if grad_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), grad_norm)
            optimizer.step()

            total_feat_loss += w_feat * feat_loss.item()
            total_deg_loss += w_deg * deg_loss.item()
            total_logit_loss += w_logit * logit_loss.item()
            total_typ_loss += w_typ * typ_loss.item()
            total_weight_loss += w_weight * weight_loss.item()
            total_moe_loss += w_moe * moe_loss.item()
            total_loss += loss.item()

        return total_loss, total_feat_loss, total_deg_loss, total_logit_loss, total_typ_loss, total_weight_loss, total_moe_loss
           
    def reconstruction_step(self, data: Data, subgraph_count, subgraph_node, device, subgraph_id):
        print(f"Subgraph count={subgraph_count}")
        # subgraph_id = random.randint(0, subgraph_count - 1)

        subgraph_nodes = subgraph_node[subgraph_id % subgraph_count]

        if self.has_edge_feat == 1:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr_max = torch.max(edge_attr)
            edge_attr_min = torch.min(edge_attr)
            if edge_attr_max - edge_attr_min < 0.0001:
                edge_attr_max += 1
                edge_attr_min -= 1
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, None
        masked_x, masked_nodes = self.graphMasker.mask_node(subgraph=subgraph_nodes, feat=x)

        # TODO: test
        remain_edge_index, mask_edge_index, remain_edge_attr, mask_edge_attr = self.graphMasker.drop_edge(subgraph=subgraph_nodes, edge_idx=edge_index, edge_attr=edge_attr, mode=1)  
        # remain_edge_index, mask_edge_index, remain_edge_attr, mask_edge_attr = edge_index, None, edge_attr, None

        origin_z = self.encoder(masked_x, remain_edge_index, remain_edge_attr)
        projected_z = self.enc2dec(origin_z)
        # projected_z = torch.rand_like(projected_z, device=origin_z.device)
        masked_z = self.featMasker(projected_z)


        # Delete all edges related to subgraph
        remain_edge_index, mask_edge_index, remain_edge_attr, mask_edge_attr = self.graphMasker.drop_edge(subgraph=subgraph_nodes, edge_idx=edge_index, edge_attr=edge_attr, mode=0)

        subgraph_nodes = torch.tensor(subgraph_nodes).to(device)

        # Reconstruct edges across the subgraph
        deg = degree(edge_index[1].flatten(), data.num_nodes).float()
        max_deg = torch.max(deg)
        
        cur_deg = degree(remain_edge_index[1].flatten(), data.num_nodes).float()
        all_nodes = torch.arange(data.num_nodes).to(device)
        outside_nodes = all_nodes[~torch.isin(all_nodes, subgraph_nodes)]

        subgraph_z = projected_z[subgraph_nodes]
        outside_z = projected_z[outside_nodes]

        node_parts = torch.cat((torch.zeros(data.num_vars), torch.ones(data.num_constrs)), dim=0).to(device)
        subgraph_parts = node_parts[subgraph_nodes]
        outside_parts = node_parts[outside_nodes]

        print(f"node outside subgraph count = {outside_nodes.shape[0]}")
        print(f"Dropped {mask_edge_index.shape[1]/2} edges across/in the subgraph.")
        print(f"cur edge shape={remain_edge_index.shape}")
        recon_edge_cnt1 = 0
        for i, outside_node in enumerate(outside_nodes):
            outside_node_part = outside_parts[i]

            opposite_part_mask = subgraph_parts != outside_node_part
            opposite_part_indices = torch.where(opposite_part_mask)[0]
            opposite_part_z = subgraph_z[opposite_part_indices]
            opposite_part_nodes = subgraph_nodes[opposite_part_indices]

            outside_node_z = outside_z[i].repeat(len(opposite_part_indices), 1)
            concatenated_z = outside_node_z * opposite_part_z
            logit_gates, _ = self.selector(x=concatenated_z, train=False)
            logit_dispatcher = SparseDispatcher(self.expert_num, logit_gates)
            logit_inputs = logit_dispatcher.dispatch(concatenated_z)
            logit_outputs = [self.linkDecoders[j](x=logit_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            link_probs = logit_dispatcher.combine(logit_outputs)
            del logit_gates, logit_dispatcher, logit_inputs, logit_outputs

            top_indices = torch.topk(link_probs, min(deg[outside_node] - cur_deg[outside_node], opposite_part_nodes.shape[0]).long(), dim=0).indices
            top_indices = torch.squeeze(top_indices, dim=1)

            top_subgraph_nodes = opposite_part_nodes[top_indices]
            outside_nodes_repeated = torch.full_like(top_subgraph_nodes, outside_node)
            if outside_parts[i] == 0:
                tmp_edges = torch.stack([outside_nodes_repeated, top_subgraph_nodes], dim=0)
            else:
                tmp_edges = torch.stack([top_subgraph_nodes, outside_nodes_repeated], dim=0)

            concatenated_z = projected_z[tmp_edges[0, :]] * projected_z[tmp_edges[1, :]]
            # concatenated_z = torch.cat((projected_z[tmp_edges[0, :]], projected_z[tmp_edges[1, :]]), dim=1)
            tmp_gates, _ = self.selector(x=concatenated_z, train=False)
            tmp_dispatcher = SparseDispatcher(self.expert_num, tmp_gates)
            tmp_inputs = tmp_dispatcher.dispatch(concatenated_z)
            typ_outputs = [self.typDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            weight_outputs = [self.weightDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            
            tmp_typs = tmp_dispatcher.combine(typ_outputs)
            tmp_masks = tmp_typs>0.5

            tmp_weights = tmp_dispatcher.combine(weight_outputs) * (edge_attr_max - edge_attr_min) + edge_attr_min
            tmp_weights[tmp_masks] = torch.floor(tmp_weights[tmp_masks] + 0.5)
            del tmp_gates, tmp_dispatcher, tmp_inputs, typ_outputs, weight_outputs

            k = len(top_indices)
            recon_edge_cnt1 += k
            swap_edges = tmp_edges[[1, 0]]
            ori_edges = torch.empty((2, 2*k)).to(device)
            
            ori_edges[:, 0::2] = tmp_edges
            ori_edges[:, 1::2] = swap_edges
            ori_weights = torch.empty((2*k, 1)).to(device)
            ori_weights[0::2] = tmp_weights
            ori_weights[1::2] = tmp_weights

            remain_edge_index = torch.cat((remain_edge_index, ori_edges.to(torch.int64)), dim=1)
            remain_edge_attr = torch.cat((remain_edge_attr, ori_weights), dim=0)
            
        print(f"Reconstructed {recon_edge_cnt1} edges across the subgraph.")
        print(f"cur edge shape={remain_edge_index.shape}")
        # Predict degree of nodes in subgraph
        deg_gates, _ = self.selector(x=projected_z, train=False)
        deg_dispatcher = SparseDispatcher(self.expert_num, deg_gates)
        deg_inputs = deg_dispatcher.dispatch(projected_z)
        deg_outputs = [self.degreeDecoders[j](x=deg_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        print(f"max deg={max_deg}")
        deg_pred = (deg_dispatcher.combine(deg_outputs).squeeze() * max_deg + 0.5).long() 
        del deg_gates, deg_dispatcher, deg_inputs, deg_outputs
        print(f"global deg min={min(deg_pred)} max={max(deg_pred)}")

        deg_now = degree(remain_edge_index[1].flatten().to(torch.int64), data.num_nodes)
        deg_subgraph = deg_pred - deg_now
        # Reconstruct edges in the subgraph
        var_nodes = subgraph_nodes[subgraph_parts == 0]
        print(f"var node size={var_nodes.shape[0]}")
        constr_nodes = subgraph_nodes[subgraph_parts == 1]
        print(f"constr node size={constr_nodes.shape[0]}")


        print(f"deg_ori var:{torch.min(deg[var_nodes])}~{torch.max(deg[var_nodes])} constr:{torch.min(deg[constr_nodes])}~{torch.max(deg[constr_nodes])}")
        print(f"deg_now var:{torch.min(deg_now[var_nodes])}~{torch.max(deg_now[var_nodes])} constr:{torch.min(deg_now[constr_nodes])}~{torch.max(deg_now[constr_nodes])}")
        print(f"deg_pred var:{torch.min(deg_pred[var_nodes])}~{torch.max(deg_pred[var_nodes])} constr:{torch.min(deg_pred[constr_nodes])}~{torch.max(deg_pred[constr_nodes])}")
        print(f"deg_subgraph var:{torch.min(deg_subgraph[var_nodes])}~{torch.max(deg_subgraph[var_nodes])} constr:{torch.min(deg_subgraph[constr_nodes])}~{torch.max(deg_subgraph[constr_nodes])}")

        deg_subgraph = torch.max(deg_subgraph, torch.zeros_like(deg_subgraph))

        var_repeat = var_nodes.repeat_interleave(len(constr_nodes))
        constr_repeat = constr_nodes.repeat(len(var_nodes))
        pairs_z = projected_z[var_repeat] * projected_z[constr_repeat]
        # pairs_z = torch.cat((projected_z[var_repeat], projected_z[constr_repeat]), dim=1)
        link_gates, _ = self.selector(x=pairs_z, train=False)
        link_dispatcher = SparseDispatcher(self.expert_num, link_gates)
        link_inputs = link_dispatcher.dispatch(pairs_z)
        link_outputs = [self.linkDecoders[j](x=link_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        link_probs = link_dispatcher.combine(link_outputs)
        del link_gates, link_dispatcher, link_inputs, link_outputs

        link_pair = torch.stack(
            (var_repeat, 
             constr_repeat, 
             torch.squeeze(link_probs, dim=1),
             torch.zeros(var_repeat.shape[0]).to(device)
            ), dim=0)

        for var_node in var_nodes:
            link_of_node = link_pair[:, link_pair[0, :] == var_node]
            link_index = torch.arange(link_pair.shape[1]).to(device)
            link_index = link_index[link_pair[0, :] == var_node]
            _, sorted_indices = link_of_node[2, :].sort(descending=True)
            link_pair[3, link_index[sorted_indices[:deg_subgraph[var_node].long().item()]]] += 1

        for constr_node in constr_nodes:
            link_of_node = link_pair[:, link_pair[1, :] == constr_node]
            link_index = torch.arange(link_pair.shape[1]).to(device)
            link_index = link_index[link_pair[1, :] == constr_node]
            _, sorted_indices = link_of_node[2, :].sort(descending=True)
            link_pair[3, link_index[sorted_indices[:deg_subgraph[constr_node].long().item()]]] += 1

        edge_setA = link_pair[:, link_pair[3, :] == 1]
        edge_setB = link_pair[:, link_pair[3, :] == 2]
        
        _, indices = torch.sort(edge_setA[2, :], descending=True)
        edge_setC = edge_setA[:, indices]
        edge_setC = edge_setC[:, :edge_setC.shape[1]//2]
        # rand_vals = torch.rand_like(edge_setA[2, :])
        # edge_setC = edge_setA[:, rand_vals < edge_setA[2, :]]
        
        edge_setD = torch.cat((edge_setB, edge_setC), dim=1)
        print(f"Edge statistics K=1:{edge_setA.shape[1]}->{edge_setC.shape[1]} K=2:{edge_setB.shape[1]}")

        tmp_edges = edge_setD[:2, :]
        concatenated_z = projected_z[tmp_edges[0, :].long()] * projected_z[tmp_edges[1, :].long()]
        # concatenated_z = torch.cat((projected_z[tmp_edges[0, :].long()], projected_z[tmp_edges[1, :].long()]), dim=1)

        tmp_gates, _ = self.selector(x=concatenated_z, train=False)
        tmp_dispatcher = SparseDispatcher(self.expert_num, tmp_gates)
        tmp_inputs = tmp_dispatcher.dispatch(concatenated_z)
        typ_outputs = [self.typDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        weight_outputs = [self.weightDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            
        tmp_typs = tmp_dispatcher.combine(typ_outputs)
        tmp_masks = tmp_typs>0.5
        tmp_weights = tmp_dispatcher.combine(weight_outputs) * (edge_attr_max - edge_attr_min) + edge_attr_min
        tmp_weights[tmp_masks] = torch.floor(tmp_weights[tmp_masks] + 0.5)
        del tmp_gates, tmp_dispatcher, tmp_inputs, typ_outputs, weight_outputs

        k = tmp_edges.shape[1]
        print(f"Reconstructed {k} edges in the subgraph.")
        swap_edges = tmp_edges[[1, 0]]
        ori_edges = torch.empty((2, 2*k)).to(device)
        ori_edges[:, 0::2] = tmp_edges
        ori_edges[:, 1::2] = swap_edges
        ori_weights = torch.empty((2*k, 1)).to(device)
        ori_weights[0::2] = tmp_weights
        ori_weights[1::2] = tmp_weights
        del tmp_edges, swap_edges, tmp_weights

        remain_edge_index = torch.cat((remain_edge_index, ori_edges.to(torch.int64)), dim=1)
        remain_edge_attr = torch.cat((remain_edge_attr, ori_weights), dim=0)

        # Reconstruct node features
        # TODO: Decide use projected_z or masked_z
        recon_x = self.featDecoder(masked_z, remain_edge_index, remain_edge_attr)
        x[subgraph_nodes] = recon_x[subgraph_nodes]
        print(f"Reconstructed {subgraph_nodes.shape[0]} nodes in the subgraph.")

        new_data = Data(x=x, edge_index=remain_edge_index, edge_attr=remain_edge_attr)
        del x, remain_edge_index, remain_edge_attr
        new_data.rnode = data.rnode + subgraph_nodes.shape[0]
        new_data.num_vars = data.num_vars
        new_data.num_constrs = data.num_constrs
        new_data.subgraph_node = data.subgraph_node
        new_data.subgraph_count = data.subgraph_count
        del data

        new_node_flag = torch.zeros(new_data.num_vars+new_data.num_constrs)
        new_node_flag[subgraph_nodes] = 1
        return new_data, new_node_flag.tolist()
    
    def extrapolation_step(self, data: Data, subgraph_count, subgraph_node, device, subgraph_id):
        if self.has_edge_feat == 1:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_attr_max = torch.max(edge_attr)
            edge_attr_min = torch.min(edge_attr)
            if edge_attr_max - edge_attr_min < 0.0001:
                edge_attr_max += 1
                edge_attr_min -= 1
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, None

        ori_num_vars, ori_num_constrs = data.num_vars, data.num_constrs
        subgraph_nodes = subgraph_node[subgraph_id % subgraph_count]
        e_num_vars = 0
        e_num_constrs = 0
        for node in subgraph_nodes:
            if node < ori_num_vars:
                e_num_vars += 1
            else:
                e_num_constrs += 1
        scale = (ori_num_vars+ori_num_constrs+len(subgraph_nodes)) / (ori_num_vars+ori_num_constrs)
        # e_num_vars = math.floor(ori_num_vars * (scale-1))
        # e_num_constrs = math.floor(ori_num_constrs * (scale-1))

        origin_z = self.encoder(x, edge_index, edge_attr)
        scaled_z = torch.zeros((ori_num_vars+ori_num_constrs+e_num_vars+e_num_constrs, origin_z.shape[1])).to(device)
        scaled_z[:ori_num_vars] = origin_z[:ori_num_vars]
        scaled_z[ori_num_vars+e_num_vars : ori_num_vars+e_num_vars+ori_num_constrs] = origin_z[ori_num_vars : ori_num_vars+ori_num_constrs]
        
        # Sample Node Embeddings
        # TODO: modify sample method
        # scaled_z[ori_num_vars:ori_num_vars+e_num_vars] = scaled_z[torch.randint(0, ori_num_vars, (e_num_vars,)).to(device)]
        # scaled_z[ori_num_vars+e_num_vars+ori_num_constrs : ori_num_vars+e_num_vars+ori_num_constrs+e_num_constrs] = scaled_z[torch.randint(ori_num_vars+e_num_vars, ori_num_vars+e_num_vars+ori_num_constrs, (e_num_constrs,)).to(device)]
        var_pointer = ori_num_vars
        constr_pointer = ori_num_vars+e_num_vars+ori_num_constrs
        for node in subgraph_nodes:
            if node < ori_num_vars:
                scaled_z[var_pointer] = scaled_z[node]
                var_pointer += 1
            else:
                scaled_z[constr_pointer] = scaled_z[node]
                constr_pointer += 1
        

        # update edge index        
        edge_index[0, edge_index[0, :] >= ori_num_vars] += e_num_vars
        edge_index[1, edge_index[1, :] >= ori_num_vars] += e_num_vars
        # Calculate Current Degree, Make deg prediction preparation
        ori_deg = degree(edge_index[1].flatten(), num_nodes=ori_num_vars+ori_num_constrs+e_num_vars+e_num_constrs).float()
        max_deg = torch.max(ori_deg)
        
        new_node_flag = torch.zeros(ori_num_vars+e_num_vars+ori_num_constrs+e_num_constrs)
        new_node_flag[ori_num_vars:ori_num_vars+e_num_vars] = 1
        new_node_flag[ori_num_vars+e_num_vars+ori_num_constrs:ori_num_vars+e_num_vars+ori_num_constrs+e_num_constrs] = 1

        # Delete some edges, Noticed that each edge appears in scaled_edge_index twice
        # tmp_edge_index = edge_index[:, 0::2]
        # tmp_edge_attr = edge_attr[0::2]
        # concatenated_z = scaled_z[tmp_edge_index[0, :]] * scaled_z[tmp_edge_index[1, :]]
        # logit_gates, _ = self.selector(x=concatenated_z, train=False)
        # logit_dispatcher = SparseDispatcher(self.expert_num, logit_gates)
        # logit_inputs = logit_dispatcher.dispatch(concatenated_z)
        # logit_outputs = [self.linkDecoders[j](x=logit_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        # link_probs = logit_dispatcher.combine(logit_outputs)

        # edge_num = data.edge_index.shape[1] // 2
        # reserved_edge_num = math.floor(edge_num / scale)
        # top_indices = torch.topk(link_probs, k=reserved_edge_num, dim=0).indices
        # top_indices = torch.squeeze(top_indices, dim=1)

        # tmp_edge_index = tmp_edge_index[:, top_indices]
        # tmp_edge_attr = tmp_edge_attr[top_indices]
        # swap_edge_index = tmp_edge_index[[1, 0]]
        # scaled_edge_index = torch.empty((2, 2*reserved_edge_num), dtype=torch.int64).to(device)
        # scaled_edge_index[:, 0::2] = tmp_edge_index
        # scaled_edge_index[:, 1::2] = swap_edge_index
        # scaled_edge_attr = torch.empty((2*reserved_edge_num, 1)).to(device)
        # scaled_edge_attr[0::2] = tmp_edge_attr
        # scaled_edge_attr[1::2] = tmp_edge_attr
        # print(f"reserved edge shape={scaled_edge_index.shape}")
        # Delete Done

        scaled_edge_attr = edge_attr
        scaled_edge_index = edge_index

        # Recover Cross-Subgraph Edges
        projected_z = self.enc2dec(scaled_z)
        masked_z = self.featMasker(projected_z)
        
        subgraph_nodes = torch.cat([torch.arange(ori_num_vars, ori_num_vars+e_num_vars), 
                                    torch.arange(ori_num_vars+e_num_vars+ori_num_constrs, ori_num_vars+e_num_vars+ori_num_constrs+e_num_constrs)]).to(device)
        cur_deg = degree(scaled_edge_index[1].flatten().to(torch.int64), num_nodes=ori_num_vars+ori_num_constrs+e_num_vars+e_num_constrs).float()
        all_nodes = torch.arange(0, ori_num_vars+e_num_vars+ori_num_constrs+e_num_constrs).to(device)
        outside_nodes = all_nodes[~torch.isin(all_nodes, subgraph_nodes)]

        subgraph_z = projected_z[subgraph_nodes]
        outside_z = projected_z[outside_nodes]

        node_parts = torch.cat((torch.zeros(ori_num_vars+e_num_vars), torch.ones(ori_num_constrs+e_num_constrs)), dim=0).to(device)
        subgraph_parts = node_parts[subgraph_nodes]
        outside_parts = node_parts[outside_nodes]

        recon_edge_cnt1 = 0
        for i, outside_node in enumerate(outside_nodes):
            outside_node_part = outside_parts[i]

            opposite_part_mask = subgraph_parts != outside_node_part
            opposite_part_indices = torch.where(opposite_part_mask)[0]
            opposite_part_z = subgraph_z[opposite_part_indices]
            opposite_part_nodes = subgraph_nodes[opposite_part_indices]

            outside_node_z = outside_z[i].repeat(len(opposite_part_indices), 1)
            concatenated_z = outside_node_z * opposite_part_z
            logit_gates, _ = self.selector(x=concatenated_z, train=False)
            logit_dispatcher = SparseDispatcher(self.expert_num, logit_gates)
            logit_inputs = logit_dispatcher.dispatch(concatenated_z)
            logit_outputs = [self.linkDecoders[j](x=logit_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            link_probs = logit_dispatcher.combine(logit_outputs)

            # top_indices = torch.topk(link_probs, min(ori_deg[outside_node].long() - cur_deg[outside_node].long(), opposite_part_nodes.shape[0]), dim=0).indices
            top_indices = torch.topk(link_probs, min(math.floor(ori_deg[outside_node].long()*(scale-1)), opposite_part_nodes.shape[0]), dim=0).indices
            top_indices = torch.squeeze(top_indices, dim=1)

            top_subgraph_nodes = opposite_part_nodes[top_indices]
            outside_nodes_repeated = torch.full_like(top_subgraph_nodes, outside_node)
            if outside_parts[i] == 0:
                tmp_edges = torch.stack([outside_nodes_repeated, top_subgraph_nodes], dim=0)
            else:
                tmp_edges = torch.stack([top_subgraph_nodes, outside_nodes_repeated], dim=0)

            concatenated_z = projected_z[tmp_edges[0, :]] * projected_z[tmp_edges[1, :]]
            # concatenated_z = torch.cat((projected_z[tmp_edges[0, :]], projected_z[tmp_edges[1, :]]), dim=1)
            tmp_gates, _ = self.selector(x=concatenated_z, train=False)
            tmp_dispatcher = SparseDispatcher(self.expert_num, tmp_gates)
            tmp_inputs = tmp_dispatcher.dispatch(concatenated_z)
            typ_outputs = [self.typDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            weight_outputs = [self.weightDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            
            tmp_typs = tmp_dispatcher.combine(typ_outputs)
            tmp_masks = tmp_typs>0.5
            tmp_weights = tmp_dispatcher.combine(weight_outputs) * (edge_attr_max - edge_attr_min) + edge_attr_min
            tmp_weights[tmp_masks] = torch.floor(tmp_weights[tmp_masks] + 0.5)

            k = len(top_indices)
            recon_edge_cnt1 += k
            swap_edges = tmp_edges[[1, 0]]
            ori_edges = torch.empty((2, 2*k)).to(device)
            
            ori_edges[:, 0::2] = tmp_edges
            ori_edges[:, 1::2] = swap_edges
            ori_weights = torch.empty((2*k, 1)).to(device)
            ori_weights[0::2] = tmp_weights
            ori_weights[1::2] = tmp_weights

            scaled_edge_index = torch.cat((scaled_edge_index, ori_edges.to(torch.int64)), dim=1)
            scaled_edge_attr = torch.cat((scaled_edge_attr, ori_weights), dim=0)
            
        print(f"Reconstructed {recon_edge_cnt1} edges across the subgraph.")
        print(f"cur edge shape={scaled_edge_index.shape}")

        #########################################
        # Building Inside Extrapolated Subgraph #
        #########################################
        deg_gates, _ = self.selector(x=projected_z, train=False)
        deg_dispatcher = SparseDispatcher(self.expert_num, deg_gates)
        deg_inputs = deg_dispatcher.dispatch(projected_z)
        deg_outputs = [self.degreeDecoders[j](x=deg_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        print(f"max deg={max_deg}")
        deg_pred = (deg_dispatcher.combine(deg_outputs).squeeze() * max_deg + 0.5).long() 
        print(f"global deg min={min(deg_pred)} max={max(deg_pred)}")

        deg_now = degree(scaled_edge_index[1].flatten().to(torch.int64), num_nodes=ori_num_vars+ori_num_constrs+e_num_vars+e_num_constrs)
        deg_subgraph = deg_pred - deg_now
        # Reconstruct edges in the subgraph

        var_nodes = subgraph_nodes[subgraph_parts == 0]
        print(f"var node size={var_nodes.shape[0]}")
        constr_nodes = subgraph_nodes[subgraph_parts == 1]
        print(f"constr node size={constr_nodes.shape[0]}")

        print(f"deg_now var:{torch.min(deg_now[var_nodes])}~{torch.max(deg_now[var_nodes])} constr:{torch.min(deg_now[constr_nodes])}~{torch.max(deg_now[constr_nodes])}")
        print(f"deg_now_avg var:{torch.sum(deg_now[var_nodes]/var_nodes.shape[0])} constr:{torch.sum(deg_now[constr_nodes]/constr_nodes.shape[0])}")
        print(f"deg_pred var:{torch.min(deg_pred[var_nodes])}~{torch.max(deg_pred[var_nodes])} constr:{torch.min(deg_pred[constr_nodes])}~{torch.max(deg_pred[constr_nodes])}")
        print(f"deg_pred_avg var:{torch.sum(deg_pred[var_nodes]/var_nodes.shape[0])} constr:{torch.sum(deg_pred[constr_nodes]/constr_nodes.shape[0])}")
        print(f"deg_subgraph var:{torch.min(deg_subgraph[var_nodes])}~{torch.max(deg_subgraph[var_nodes])} constr:{torch.min(deg_subgraph[constr_nodes])}~{torch.max(deg_subgraph[constr_nodes])}")

        deg_subgraph = torch.max(deg_subgraph, torch.zeros_like(deg_subgraph))

        var_repeat = var_nodes.repeat_interleave(len(constr_nodes))
        constr_repeat = constr_nodes.repeat(len(var_nodes))
        pairs_z = projected_z[var_repeat] * projected_z[constr_repeat]
        # pairs_z = torch.cat((projected_z[var_repeat], projected_z[constr_repeat]), dim=1)
        link_gates, _ = self.selector(x=pairs_z, train=False)
        link_dispatcher = SparseDispatcher(self.expert_num, link_gates)
        link_inputs = link_dispatcher.dispatch(pairs_z)
        link_outputs = [self.linkDecoders[j](x=link_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        link_probs = link_dispatcher.combine(link_outputs)

        link_pair = torch.stack(
            (var_repeat, 
             constr_repeat, 
             torch.squeeze(link_probs, dim=1),
             torch.zeros(var_repeat.shape[0]).to(device)
            ), dim=0)

        for var_node in var_nodes:
            link_of_node = link_pair[:, link_pair[0, :] == var_node]
            link_index = torch.arange(link_pair.shape[1]).to(device)
            link_index = link_index[link_pair[0, :] == var_node]
            _, sorted_indices = link_of_node[2, :].sort(descending=True)
            link_pair[3, link_index[sorted_indices[:deg_subgraph[var_node].long().item()]]] += 1

        for constr_node in constr_nodes:
            link_of_node = link_pair[:, link_pair[1, :] == constr_node]
            link_index = torch.arange(link_pair.shape[1]).to(device)
            link_index = link_index[link_pair[1, :] == constr_node]
            _, sorted_indices = link_of_node[2, :].sort(descending=True)
            link_pair[3, link_index[sorted_indices[:deg_subgraph[constr_node].long().item()]]] += 1

        edge_setA = link_pair[:, link_pair[3, :] == 1]
        edge_setB = link_pair[:, link_pair[3, :] == 2]
        
        _, indices = torch.sort(edge_setA[2, :], descending=True)
        edge_setC = edge_setA[:, indices]
        edge_setC = edge_setC[:, :edge_setC.shape[1]//2]
        # rand_vals = torch.rand_like(edge_setA[2, :])
        # edge_setC = edge_setA[:, rand_vals < edge_setA[2, :]]
        
        edge_setD = torch.cat((edge_setB, edge_setC), dim=1)
        print(f"Edge statistics K=1:{edge_setA.shape[1]}->{edge_setC.shape[1]} K=2:{edge_setB.shape[1]}")

        tmp_edges = edge_setD[:2, :]
        concatenated_z = projected_z[tmp_edges[0, :].long()] * projected_z[tmp_edges[1, :].long()]


        tmp_gates, _ = self.selector(x=concatenated_z, train=False)
        tmp_dispatcher = SparseDispatcher(self.expert_num, tmp_gates)
        tmp_inputs = tmp_dispatcher.dispatch(concatenated_z)
        typ_outputs = [self.typDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
        weight_outputs = [self.weightDecoders[j](x=tmp_inputs[j], sigmoid=True) for j in range(self.expert_num)]
            
        tmp_typs = tmp_dispatcher.combine(typ_outputs)
        tmp_masks = tmp_typs>0.5
        tmp_weights = tmp_dispatcher.combine(weight_outputs) * (edge_attr_max - edge_attr_min) + edge_attr_min
        tmp_weights[tmp_masks] = torch.floor(tmp_weights[tmp_masks] + 0.5)

        k = tmp_edges.shape[1]
        print(f"Reconstructed {k} edges in the subgraph.")
        swap_edges = tmp_edges[[1, 0]]
        ori_edges = torch.empty((2, 2*k)).to(device)
        ori_edges[:, 0::2] = tmp_edges
        ori_edges[:, 1::2] = swap_edges
        ori_weights = torch.empty((2*k, 1)).to(device)
        ori_weights[0::2] = tmp_weights
        ori_weights[1::2] = tmp_weights

        scaled_edge_index = torch.cat((scaled_edge_index, ori_edges.to(torch.int64)), dim=1)
        scaled_edge_attr = torch.cat((scaled_edge_attr, ori_weights), dim=0)

        # Using GNN Decoder
        scaled_x = self.featDecoder(masked_z, scaled_edge_index, scaled_edge_attr)
        scaled_x[:ori_num_vars] = x[:ori_num_vars]
        scaled_x[ori_num_vars+e_num_vars:ori_num_vars+e_num_vars+ori_num_constrs] = x[ori_num_vars:ori_num_vars+ori_num_constrs]

        new_data = Data(x=scaled_x, edge_index=scaled_edge_index, edge_attr=scaled_edge_attr)
        new_data.num_vars = ori_num_vars + e_num_vars
        new_data.num_constrs = ori_num_constrs + e_num_constrs

        return new_data, new_node_flag.tolist()