import numpy as np
import argparse
import pickle
import os
import time
import random
from gurobipy import *
from functools import cmp_to_key
from model.gbdt_regressor import GradientBoostingRegressor

import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_metric_learning import losses

from model.graphcnn import GNNPolicy

class pair: 
    def __init__(self): 
        self.site = 0
        self.loss = 0

def cmp(a, b): # 定义cmp
    if a.loss < b.loss: 
        return -1 # 不调换
    else:
        return 1 # 调换

def cmp2(a, b): # 定义cmp
    if a.loss > b.loss: 
        return -1 # 不调换
    else:
        return 1 # 调换

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type, now_sol, now_col):
    '''
    函数说明：
    根据传入的问题实例，使用Gurobi求解器进行求解。

    参数说明：
    - n: 问题实例的决策变量数量。
    - m: 问题实例的约束数量。
    - k: k[i]表示第i条约束的决策变量数量。
    - site: site[i][j]表示第i个约束的第j个决策变量是哪个决策变量。
    - value: value[i][j]表示第i个约束的第j个决策变量的系数。
    - constraint: constraint[i]表示第i个约束右侧的数。
    - constraint_type: constraint_type[i]表示第i个约束的类型，1表示<=，2表示>=
    - coefficient: coefficient[i]表示第i个决策变量在目标函数中的系数。
    - time_limit: 最大求解时间。
    - obj_type: 问题是最大化问题还是最小化问题。
    '''
    #获得起始时间
    begin_time = time.time()
    #定义求解模型
    model = Model("Gurobi")
    #设定变量映射
    site_to_new = {}
    new_to_site = {}
    new_num = 0
    x = []
    for i in range(n):
        if(now_col[i] == 1):
            site_to_new[i] = new_num
            new_to_site[new_num] = i
            new_num += 1
            if(value_type[i] == 'B'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
            elif(value_type[i] == 'C'):
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
            else:
                x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
    
    #设定目标函数和优化目标（最大化/最小化）
    coeff = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
        else:
            coeff += now_sol[i] * coefficient[i]
    if(obj_type == 'maximize'):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    #添加m条约束
    for i in range(m):
        constr = 0
        flag = 0
        for j in range(k[i]):
            if(now_col[site[i][j]] == 1):
                constr += x[site_to_new[site[i][j]]] * value[i][j]
                flag = 1
            else:
                constr += now_sol[site[i][j]] * value[i][j]

        if(flag == 1):
            if(constraint_type[i] == 1):
                model.addConstr(constr <= constraint[i])
            elif(constraint_type[i] == 2):
                model.addConstr(constr >= constraint[i])
            else:
                model.addConstr(constr == constraint[i])
        else:
            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    print(now_col)
            else:
                if(constr < constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    print(now_col)
    #设定最大求解时间
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    #优化求解
    model.optimize()
    try:
        new_sol = []
        for i in range(n):
            if(now_col[i] == 0):
                new_sol.append(now_sol[i])
            else:
                if(value_type[i] == 'C'):
                    new_sol.append(x[site_to_new[i]].X)
                else:
                    new_sol.append((int)(x[site_to_new[i]].X))
            
        return new_sol, model.ObjVal
    except:
        return -1, -1

def random_generate_blocks(n, m, k, site, values, loss, fix, rate, predict, nowX):
    parts = []
    score = []
    number = []
    parts.append(np.zeros(n))
    score.append(0)
    number.append(0)

    fix_num = (int)(fix * n)
    
    fix_color = np.zeros(n)
    for i in range(fix_num):
        fix_color[values[i].site] = 1
    
    now_site = 0
    for i in range(m):
        new_num = number[now_site]
        for j in range(k[i]):
            if(parts[now_site][site[i][j]] == 0):
                new_num += 1
        if(new_num > (int)(rate * n)):
            now_site += 1
            parts.append(np.zeros(n))
            score.append(0)
            number.append(0)
        
        for j in range(k[i]):
            if(parts[now_site][site[i][j]] == 0):
                parts[now_site][site[i][j]] = 1
                score[now_site] += loss[site[i][j]] * abs(nowX[site[i][j]] - predict[site[i][j]])
                number[now_site] += 1
    
    return(parts, score, number)

def cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data):
    parts = []
    score = []
    number = []
    for i in range(4):
        parts.append(np.zeros(n))
        score.append(0)
        number.append(0)
    
    pairs = []
    for i in range(n):
        pairs.append(pair())
        pairs[i].site= i
        pairs[i].loss = loss[i] * abs(nowX[i] - predict[i])
    pairs.sort(key = cmp_to_key(cmp2))  
    
    now_tree = random.randint(0, 29)
    max_num = n * rate

    root = GBDT.trees[now_tree].root
    left = GBDT.trees[now_tree].root.left
    right = GBDT.trees[now_tree].root.right
    for i in range(n):
        now_site = pairs[i].site
        now_score = pairs[i].loss
        if(data[now_site][root.feature] < root.split):
            if(data[now_site][left.feature] < left.split):
                if(number[0] <= max_num):
                    parts[0][now_site] = 1
                    number[0] += 1
                    score[0] += now_score
            else:
                if(number[1] <= max_num):
                    parts[1][now_site] = 1
                    number[1] += 1
                    score[1] += now_score
        else:
            if(data[now_site][right.feature] < right.split):
                if(number[2] <= max_num):
                    parts[2][now_site] = 1
                    number[2] += 1
                    score[2] += now_score
            else:
                if(number[3] <= max_num):
                    parts[3][now_site] = 1
                    number[3] += 1
                    score[3] += now_score
    return(parts, score, number)

def cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, solA, blockA, solB, blockB, set_time, lower_bound, upper_bound, value_type):
    crossX = np.zeros(n)
    
    for i in range(n):
        if(blockA[i] == 1):
            crossX[i] = solA[i]
        else:
            crossX[i] = solB[i]
    
    color = np.zeros(n)
    add_num = 0
    for j in range(m):
        constr = 0
        flag = 0
        for l in range(k[j]):
            if(color[site[j][l]] == 1):
                flag = 1
            else:
                constr += crossX[site[j][l]] * value[j][l]

        if(flag == 0):
            if(constraint_type[j] == 1):
                if(constr > constraint[j]):
                    for l in range(k[j]):
                        if(color[site[j][l]] == 0):
                            color[site[j][l]] = 1
                            add_num += 1
            else:
                if(constr < constraint[j]):
                    for l in range(k[j]):
                        if(color[site[j][l]] == 0):
                            color[site[j][l]] = 1
                            add_num += 1
    if(add_num / n <= rate):
        newcrossX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, set_time, obj_type, lower_bound, upper_bound, value_type, crossX, color)
        return newcrossX, newVal
    else:
        return -1, -1

def read_lp(now_lp_file : str):
    #创建模型和最优解读入
    model = read(now_lp_file)

    value_to_num = {}
    value_to_type = {}
    value_num = 0
    #n表示决策变量个数
    #m表示约束数量
    #k[i]表示第i条约束中决策变量数量
    #site[i][j]表示第i个约束的第j个决策变量是哪个决策变量
    #value[i][j]表示第i个约束的第j个决策变量的系数
    #constraint[i]表示第i个约束右侧的数
    #constraint_type[i]表示第i个约束的类型，1表示<，2表示>，3表示=
    #coefficient[i]表示第i个决策变量在目标函数中的系数
    n = model.NumVars
    m = model.NumConstrs
    print(n, m)
    k = []
    site = []
    value = []
    constraint = []
    constraint_type = []
    for cnstr in model.getConstrs():
        if(cnstr.Sense == '<'):
            constraint_type.append(1)
        elif(cnstr.Sense == '>'):
            constraint_type.append(2) 
        else:
            constraint_type.append(3) 
        
        constraint.append(cnstr.RHS)


        now_site = []
        now_value = []
        row = model.getRow(cnstr)
        k.append(row.size())
        for i in range(row.size()):
            if(row.getVar(i).VarName not in value_to_num.keys()):
                value_to_num[row.getVar(i).VarName] = value_num
                value_num += 1
            now_site.append(value_to_num[row.getVar(i).VarName])
            now_value.append(row.getCoeff(i))
        site.append(now_site)
        value.append(now_value)

    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}

    now_solution = []
    for i in range(n):
        now_solution.append(0)

    for val in model.getVars():
        if(val.VarName not in value_to_num.keys()):
            value_to_num[val.VarName] = value_num
            value_num += 1
        coefficient[value_to_num[val.VarName]] = val.Obj
        lower_bound[value_to_num[val.VarName]] = val.LB
        upper_bound[value_to_num[val.VarName]] = val.UB
        value_type[value_to_num[val.VarName]] = val.Vtype

    #1最小化，-1最大化
    obj_type = model.ModelSense
    if(obj_type == 1):
        obj_type = 'minimize'
    else:
        obj_type = 'maximize'


    #初始特征编码！
    variable_features = []
    constraint_features = []
    edge_indices = [[], []] 
    edge_features = []

    for i in range(n):
        now_variable_features = []
        now_variable_features.append(coefficient[i])
        now_variable_features.append(lower_bound[i])
        now_variable_features.append(upper_bound[i])
        if(value_type[i] == 'C'):
            now_variable_features.append(0)
        else:
            now_variable_features.append(1)
        now_variable_features.append(random.random())
        variable_features.append(now_variable_features)
    
    for i in range(m):
        now_constraint_features = []
        now_constraint_features.append(constraint[i])
        now_constraint_features.append(constraint_type[i])
        now_constraint_features.append(random.random())
        constraint_features.append(now_constraint_features)
    
    for i in range(m):
        for j in range(k[i]):
            edge_indices[0].append(i)
            edge_indices[1].append(site[i][j])
            edge_features.append([value[i][j]])
    
    problem = [obj_type, n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type]
    pair = [variable_features, constraint_features, edge_indices, edge_features, [], []]
    return problem, pair

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


def optimize(id : int,
             problem : str,
             difficulty : str,
             fix : float,
             set_time : int,
             rate : float):
    begin_time = time.time()

    now_lp_file = '/home/sharing/disk1/yehuigen/NewData/_latest_datasets/0_generated_instances/generate_' + problem + '/' + problem + '_' + difficulty + '_instance/LP/' + problem + '_' + difficulty + '_instance_' + str(id) + '.lp'

    if(os.path.exists(now_lp_file) == False):
        print("No problem file!")
    matrix, pairs = read_lp(now_lp_file)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = './trained_model_' + problem + '_' + difficulty + '.pkl'
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    logits = policy(
        torch.FloatTensor(pairs[1]).to(device),
        torch.LongTensor(pairs[2]).to(device),
        torch.FloatTensor(pairs[3]).to(device),
        torch.FloatTensor(pairs[0]).to(device),
    )
    #print(logits)
 
    data = logits.tolist()

    obj_type = matrix[0]
    n = matrix[1]
    m = matrix[2]
    k = matrix[3]
    site = matrix[4]
    value = matrix[5]
    constraint = matrix[6]
    constraint_type = matrix[7]
    coefficient = matrix[8]
    lower_bound = matrix[9]
    upper_bound = matrix[10]
    value_type = matrix[11]


    if(os.path.exists('./GBDT_' + problem + '_' + difficulty + '.pickle') == False):
        print("No problem file!")

    with open('./GBDT_' + problem + '_' + difficulty + '.pickle', "rb") as f:
        GBDT = pickle.load(f)[0]

    predict = GBDT.predict(np.array(data))
    loss =  GBDT.calc(np.array(data))

    values = []
    for i in range(n):
        values.append(pair())
        values[i].site= i
        values[i].loss = loss[i]
    
    random.shuffle(values)
    values.sort(key = cmp_to_key(cmp))  

    set_rate = 1
    
    for turn in range(10):
        obj = (int)(n * (1 - set_rate * rate))

        solution = []
        color = []
        for i in range(n):
            solution.append(0)
            color.append(0)

        for i in range(n):
            now_site = values[i].site
            if(i < obj):
                if(predict[now_site] < 0.5):
                    solution[now_site] = 0
                else:
                    solution[now_site] = 1
            else:
                color[now_site] = 1

        for i in range(m):
            constr = 0
            flag = 0
            for j in range(k[i]):
                if(color[site[i][j]] == 1):
                    flag = 1
                else:
                    constr += solution[site[i][j]] * value[i][j]

            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
                            constr -= solution[site[i][j]] * value[i][j]
            else:
                if(constr + flag < constraint[i]):
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
                            break
        print(obj / n)
        if(obj / n + rate >= 1):
            break
        else:
            set_rate -= 0.1

    #初始解
    ansTime = []
    ansVal = []
    nowX, nowVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, (0.5 * set_time), obj_type, lower_bound, upper_bound, value_type, solution, color) 
    #print("nowX", nowX)
    #print("nowVal", nowVal)
    ansTime.append(time.time() - begin_time)
    ansVal.append(nowVal)

    
    random_flag = 0
    while(time.time() - begin_time < set_time):
        if(random_flag == 1):
            turnX = []
            for i in range(n):
                turnX.append(0)
            if(obj_type == 'maximize'):
                turnVal = 0
            else:
                turnVal = 1e9
            block_list, score, _ = random_generate_blocks(n, m, k, site, values, loss, fix, rate, predict, nowX)
            neibor_num = len(block_list)
            for turn in range(int(1 / rate)):
                i = 0
                now_loss = 0
                for j in range(3):
                    now_site = random.randint(0, neibor_num - 1)
                    if(score[now_site] > now_loss):
                        now_loss = score[now_site]
                        i = now_site
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(max_time, 0.2 * set_time), obj_type, lower_bound, upper_bound, value_type, nowX, block_list[i])
                if(newVal == -1):
                    continue
                if(obj_type == 'maximize'):
                    if(newVal > turnVal):
                        turnVal = newVal
                        for j in range(n):
                            turnX[j] = newX[j]
                else:
                    if(newVal < turnVal):
                        turnVal = newVal
                        for j in range(n):
                            turnX[j] = newX[j]
            if(obj_type == 'maximize'):
                if(turnVal == 0):
                    continue
            else:
                if(turnVal == 1e9):
                    continue

            for i in range(n):
                nowX[i] = turnX[i]
            nowVal = turnVal
            ansTime.append(time.time() - begin_time)
            ansVal.append(nowVal)
        else:
            turnX = []
            turnVal = []
            block_list, _, _ = cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data)
            for i in range(4):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(max_time, 0.2 * set_time), obj_type, lower_bound, upper_bound, value_type, nowX, block_list[i])
                if(newVal == -1):
                    continue
                turnX.append(newX)
                turnVal.append(newVal)
            
            #cross
            if(len(turnX) == 4):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[0], block_list[0], turnX[1], block_list[1], min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)

                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[2], block_list[2], turnX[3], block_list[3],  min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
            if(len(turnX) == 6):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break

                block_list.append(np.zeros(n, int))
                for i in range(n):
                    if(block_list[0][i] == 1 or block_list[1][i] == 1):
                        block_list[4][i] = 1
                block_list.append(np.zeros(n, int))
                for i in range(n):
                    if(block_list[2][i] == 1 or block_list[3][i] == 1):
                        block_list[5][i] = 1
                
                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[4], block_list[4], turnX[5], block_list[5], min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newX != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
            
            for i in range(len(turnVal)):
                if(obj_type == 'maximize'):
                    if(turnVal[i] > nowVal):
                        nowVal = turnVal[i]
                        for j in range(n):
                            nowX[j] = turnX[i][j]
                else:
                    if(turnVal[i] < nowVal):
                        nowVal = turnVal[i]
                        for j in range(n):
                            nowX[j] = turnX[i][j]
            
            ansTime.append(time.time() - begin_time)
            ansVal.append(nowVal)
            print(rate, "SC")
    print(ansTime)
    print(ansVal)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type = int, default = 0, help = 'The number of instances.')
    parser.add_argument("--problem", type = str, default = 'IS', help = 'The number of instances.')
    parser.add_argument("--difficulty", type = str, default = 'easy', help = 'The number of instances.')
    parser.add_argument("--fix", type = float, default = 0.6, help = 'time.')
    parser.add_argument("--set_time", type = int, default = 100, help = 'set_time.')
    parser.add_argument("--rate", type = float, default = 0.4, help = 'sub rate.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    optimize(**vars(args))