import torch
from torch_geometric.data import Data, Dataset, download_url
import os
from torch_geometric.loader import DataLoader
import argparse
import gurobipy as gp
from gurobipy import GRB
import pickle
import math
import random
from torch_geometric.utils import to_networkx
import community.community_louvain as community_louvain

def parse_args():
    # It loads MILP problems in .mps format and outputs pickled Data class of Torch Geometric.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--time_limit", type=int, default=600)
    parser.add_argument("--subgraph_limit", type=int, default=50)
    parser.add_argument('--mode', choices=["model2data", "data2model", "validate"])
    parser.add_argument('--type', choices=["slack", "direct"])
    return parser.parse_args()

def preprocess(data: Data):

    for i in range(data.num_vars):
        if data.x[i, 1] == 1:
            if data.x[i, 4] >= 0.5:
                data.x[i, 4] = 1
            else:
                data.x[i, 4] = 0
        elif data.x[i, 2] == 1:
            data.x[i, 4] = math.floor(data.x[i, 4]+0.5)
        elif data.x[i, 3] == 1:
            pass
        else:
            raise ValueError
    return data

def postprocess(data: Data, device, new_node_list, type):
    # Var
    nvars = data.num_vars
    nconstrs = data.num_constrs
    data.x[:data.num_vars, 0] = 0

    var_type_cnt = 0
    var_sample = []
    for i in range(data.num_vars):
        if new_node_list[i] == 1:
            continue
        flg = 1
        for j in range(var_type_cnt):
            if var_sample[j][1] == data.x[i, 1] and var_sample[j][2] == data.x[i, 2] and var_sample[j][3] == data.x[i, 3] and var_sample[j][6] == data.x[i, 6] and var_sample[j][7] == data.x[i, 7] and var_sample[j][8] == data.x[i, 8] and var_sample[j][9] == data.x[i, 9]:
                flg = 0
                break 

        if flg == 1:
            var_type_cnt += 1
            var_sample.append(data.x[i, :])

    for i in range(data.num_vars):
        if new_node_list[i] == 0:
            continue
        typ_id = 0
        typ_diff = 100000000
        for j in range(var_type_cnt):
            tmp_diff = 0
            # Var Type diff
            for k in range(1, 4):
                tmp_diff += 100*abs(var_sample[j][k] - data.x[i, k])
            # Bounded diff
            for k in range(6, 8):
                tmp_diff += 10*abs(var_sample[j][k] - data.x[i, k])
            # LB & UB diff
            for k in range(8, 10):
                tmp_diff += 10*abs(var_sample[j][k] - data.x[i, k])   
            if tmp_diff < typ_diff:
                typ_diff = tmp_diff
                typ_id = j
        data.x[i, 1:4] = var_sample[typ_id][1:4]
        # TODO: experimental
        data.x[i, 5] = var_sample[typ_id][5]
        data.x[i, 6:10] = var_sample[typ_id][6:10]

        if type == "slack":
            if data.x[i, 1] == 1:
                if data.x[i, 4] >= 0.5:
                    data.x[i, 4] = 1
                else:
                    data.x[i, 4] = 0
            elif data.x[i, 2] == 1:
                if data.x[i, 6] == 1:
                    data.x[i, 4] = max(data.x[i, 8], data.x[i, 4])
                if data.x[i, 7] == 1:
                    data.x[i, 4] = min(data.x[i, 9], data.x[i, 4])
                data.x[i, 4] = math.floor(data.x[i, 4]+0.5)
            elif data.x[i, 3] == 1:
                if data.x[i, 6] == 1:
                    data.x[i, 4] = max(data.x[i, 8], data.x[i, 4])
                if data.x[i, 7] == 1:
                    data.x[i, 4] = min(data.x[i, 9], data.x[i, 4])
            else:
                raise ValueError


    data.x[nvars: nvars+nconstrs, 0] = 1
    data.x[nvars: nvars+nconstrs, 1:10] = 0
    constr_type_cnt = 0
    constr_sample = []

    if type == "slack":
        data.x[nvars: nvars+nconstrs, 10] = torch.max(data.x[nvars: nvars+nconstrs, 10], torch.zeros_like(data.x[nvars: nvars+nconstrs, 10]))
    for i in range(nvars, nvars+nconstrs):
        if new_node_list[i] == 1:
            continue
        flg = 1
        for j in range(constr_type_cnt):
            if constr_sample[j][10] == data.x[i, 10]:
                flg = 0
                break 
        if flg == 1:
            constr_type_cnt += 1
            constr_sample.append(data.x[i, :])
    
    for i in range(nvars, nvars+nconstrs):
        if new_node_list[i] == 0:
            continue
        if data.x[i, 12] > 0.5:
            data.x[i, 10] = math.floor(data.x[i, 10] + 0.5)
        typ_id = 0
        typ_diff = 100000000
        for j in range(constr_type_cnt):
            tmp_diff = abs(constr_sample[j][10] - data.x[i, 10])
            if tmp_diff < typ_diff:
                typ_diff = tmp_diff
                typ_id = j
        data.x[i, 10] = constr_sample[typ_id][10]

    return data

def validate(data: Data):
    print(data)
    nvars = data.num_vars
    nconstrs = data.num_constrs
    print(f"Var num={nvars}, Constr num={nconstrs}")
    binary_count, integer_count, continous_count = 0, 0, 0
    for i in range(nvars+nconstrs):
        if i < nvars:
            assert data.x[i, 0] == 0
            assert data.x[i, 1] + data.x[i, 2] + data.x[i, 3] == 1
            if data.x[i, 1] == 1:
                binary_count += 1
                assert data.x[i, 4] == 0 or data.x[i, 4] == 1
            elif data.x[i, 2] == 1:
                integer_count += 1
                if data.x[i, 4].long() != data.x[i, 4]:
                    print(data.x[i, 4].long(), data.x[i, 4])
                    assert data.x[i, 4].long() == data.x[i, 4]
            elif data.x[i, 3] == 1:
                continous_count += 1
            else:
                raise ValueError(f"Error at position {i}")

            if data.x[i, 6] == 1:
                assert data.x[i, 8] <= data.x[i, 4]
            if data.x[i, 7] == 1:
                assert data.x[i, 9] >= data.x[i, 4]
        else:  
            assert data.x[i, 0] == 1
    assert not torch.isinf(data.x).any()
    print(f"binary count={binary_count}, integer count={integer_count}, continous count={continous_count}")
            

def transform_to_standard(model):
    if model.ModelSense != GRB.MAXIMIZE:
        for v in model.getVars():
            v.Obj *= -1
        model.ModelSense = GRB.MAXIMIZE

    model.update()
    new_constraints = []

    for constr in model.getConstrs():
        expr = model.getRow(constr)
        if constr.Sense == '<':
            pass
        elif constr.Sense == '>':
            new_constraints.append((-expr, -constr.RHS))
        elif constr.Sense == '=':
            new_constraints.append((expr, constr.RHS))
            new_constraints.append((-expr, -constr.RHS))
        else:
            raise TypeError("Unexpected Sense")

    model.update()
    for constr in model.getConstrs():
        if constr.Sense == '>' or constr.Sense == '=':
            model.remove(constr)

    model.update()
    for expr, rhs in new_constraints:
        model.addLConstr(expr, GRB.LESS_EQUAL, rhs)
    model.update()

    return model

def get_var_features(variable, typ):
    # Extract features like variable type, objective coefficient
    var_feat = [0 for i in range(14)]
    if variable.VType == 'B':
        var_feat[1] = 1
    elif variable.VType == 'I':
        var_feat[2] = 1
    elif variable.VType == 'C':
        var_feat[3] = 1
    else:
        raise TypeError("Unexpected variable type.")

    if typ == 1:
        var_feat[4] = variable.X
    var_feat[5] = variable.Obj

    if variable.LB != -gp.GRB.INFINITY and not math.isinf(variable.LB):
        var_feat[6] = 1
        var_feat[8] = variable.LB

    if variable.UB != gp.GRB.INFINITY and not math.isinf(variable.UB):
        var_feat[7] = 1
        var_feat[9] = variable.UB

    if abs(var_feat[5] - math.floor(var_feat[5])) < 0.00001:
        var_feat[11] = 1
    var_feat[13] = random.random()
    return var_feat

def get_con_features(constraint):
    con_feat = [0 for i in range(14)]
    con_feat[0] = 1
    con_feat[10] = constraint.RHS
    con_feat[13] = random.random()
    return con_feat

def GRBModel_to_Data(model, type):
    variables = model.getVars()
    constraints = model.getConstrs()

    tot_edge = 0
    for i, constr in enumerate(constraints):
        expr = model.getRow(constr)
        sz = expr.size()
        tot_edge += sz
    if(len(variables) > 50000 or tot_edge > 600000):
        return -1, -1

    var_index_map = {var: idx for idx, var in enumerate(variables)}

    if type == "slack":
        var_nodes = torch.tensor([get_var_features(v, 1) for v in variables], dtype=torch.float)
    else:
        var_nodes = torch.tensor([get_var_features(v, 0) for v in variables], dtype=torch.float)
    con_nodes = torch.tensor([get_con_features(c) for c in constraints], dtype=torch.float)
    edges = []
    edge_features = []
    for i, constr in enumerate(constraints):
        expr = model.getRow(constr)
        sz = expr.size()
        lhs_value = 0
        for j in range(sz):
            coef, var = expr.getCoeff(j), expr.getVar(j)
            if type == "slack":
                lhs_value += coef * var.X
            edges.append([var_index_map[var], len(variables) + i])
            edge_features.append(coef)
            edges.append([len(variables) + i, var_index_map[var]])
            edge_features.append(coef)
        con_nodes[i][10] -= lhs_value
        if abs(con_nodes[i][10] - math.floor(con_nodes[i][10])) < 0.000001:
            con_nodes[i][12] = 1

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)  

    data = Data(x=torch.cat([var_nodes, con_nodes]), edge_index=edge_index, edge_attr=edge_attr)
    data.num_vars = model.NumVars
    data.num_constrs = model.NumConstrs
    return data, 1

def Data_to_GRBModel(data, typ):
    model = gp.Model()
    eps = 0.000001

    var_deg = [0 for i in range(data.num_vars)]
    for i in range(data.edge_index.size(1)):
        if i % 2 == 1:
            continue
        var_idx = data.edge_index[0, i].item()
        if abs(data.edge_attr[i].item()) > eps:
            var_deg[var_idx] += 1

    var_nodes = data.x[:data.num_vars]
    con_nodes = data.x[data.num_vars:]

    variables = []
    for i, node in enumerate(var_nodes):
        var_feat = node.tolist()
        if var_feat[1] == 1:
            var_type = GRB.BINARY
        elif var_feat[2] == 1:
            var_type = GRB.INTEGER
        elif var_feat[3] == 1:
            var_type = GRB.CONTINUOUS
        else:
            raise TypeError("Cannot get var type.")
        
        if var_feat[6] == 1:
            LB = var_feat[8]
        else:
            LB = -GRB.INFINITY
        if var_feat[7] == 1:
            UB = var_feat[9]
        else:
            UB = GRB.INFINITY

        if var_feat[11] > 0.5:
            var_feat[5] = math.floor(var_feat[5] + 0.5)
        if var_deg[i] == 0:
            variables.append(None)
        else:
            var = model.addVar(vtype=var_type, obj=var_feat[5], lb=LB, ub=UB)
            variables.append(var)
    model.update()

    # Create a mapping from constraints to edges
    con_edges = {}
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


    # Create constraints
    for i, node in enumerate(con_nodes):
        con_feat = node.tolist()
        
        expr = gp.LinExpr()
        lhs_value = 0
        flag = 0
        if i in con_edges:
            for var_idx, coef in con_edges[i]:
                flag = 1
                expr.add(variables[var_idx], coef)
                if typ == "slack":
                    lhs_value += coef * var_nodes[var_idx][4].item()

        if con_feat[12] > 0.5:
            con_feat[10] = math.floor(con_feat[10] + 0.5)
        rhs_value = con_feat[10] + lhs_value

        if flag == 1:
            model.addConstr(expr <= rhs_value)
        
    model.ModelSense = GRB.MAXIMIZE
    model.update()

    return model

def print_model(model):
    print("Objective:")
    print(model.getObjective())

    # Print Variables
    print("\nVariables:")
    for v in model.getVars():
        print(f"{v.VarName}: {v.X}")

    # Print Constraints
    print("\nConstraints:")
    for c in model.getConstrs():
        constraint_expr = model.getRow(c)
        print(f"{c.ConstrName}: {constraint_expr} {c.sense} {c.RHS}")

if __name__ == "__main__":
    args = parse_args()
    filelist = os.listdir(args.input_dir)

    num_flag = 0
    for filename in filelist:
        if(num_flag >= 100):
            break
        if args.mode == "model2data":
            model = gp.read(os.path.join(args.input_dir, filename))
            model = transform_to_standard(model)

            if args.type == "slack":
                model.setParam('TimeLimit', args.time_limit)
                model.optimize()

            data, flag = GRBModel_to_Data(model, args.type)
            if(flag == -1):
                print("Too large!")
                continue
            num_flag += 1

            G = to_networkx(data, to_undirected=True)
            partition = community_louvain.best_partition(G)
            partit = []
            for i in range(data.num_nodes):
                partit.append(partition[i])

            max_id = max(partit)
            data.subgraph_node = []
            data.subgraph_count = 0

            node_lists = [[] for i in range(max_id+1)]
            for i in range(data.num_nodes):
                node_lists[partit[i]].append(i)

            for i in range(max_id+1):
                if len(node_lists[i]) > args.subgraph_limit:
                    data.subgraph_node.append(node_lists[i])
                    data.subgraph_count += 1

            data = preprocess(data)
            nlist = filename.split(".")
            tmpname = ""
            for i in range(len(nlist) - 1):
                tmpname = tmpname + nlist[i]
            file = open(os.path.join(args.output_dir, tmpname+".pkl"), 'wb')

            pickle.dump(data, file)
            file.close()
        elif args.mode == "data2model":
            print(f"Solving {filename}...")
            file = open(os.path.join(args.input_dir, filename),'rb')
            data = pickle.load(file)
            file.close()
            print(data)
            model = Data_to_GRBModel(data, args.type)
            model.write(os.path.join(args.output_dir, filename[:-4]+".lp"))

            model.setParam('TimeLimit', args.time_limit)
            model.optimize()
        else:
            file = open(os.path.join(args.input_dir, filename),'rb')
            data = pickle.load(file)
            file.close()
            print(f"Checking {filename}...")
            validate(data)
            print(f"{filename} passed.")