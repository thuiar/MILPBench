from functools import cmp_to_key
from pathlib import Path
from gurobipy import *
import numpy as np
import argparse
import pickle
import random
import time
import os

class pair: 
    def __init__(self): 
        self.x = 0
        self.y = 0
        self.val = 0

def cmp(a, b): # 定义cmp
    if a.val > b.val: 
        return -1 # 不调换
    else:
        return 1 # 调换

def find(x, father):
    if(father[x] == x):
        return x
    else:
        newx = find(father[x], father)
        father[x] = newx
        return(newx)

def generate_pair(
    problem : str,
    difficulty : str,
    ban : str,
    max_num : int
):
    #处理读入路径
    lp_files = [str(path) for path in Path('/home/sharing/disk1/yehuigen/_latest_datasets/MIPcc23-Dataset/vary_rhs/series_3/LP').glob("*.lp")]
    solution_files = [str(path) for path in Path('/home/sharing/disk1/yehuigen/_latest_datasets/MIPcc23-Dataset/vary_rhs/series_3/Pickle').glob("*.pickle")]
    
    pre_len = len('/home/sharing/disk1/yehuigen/_latest_datasets/MIPcc23-Dataset/vary_rhs/series_3/LP/rhs_s3_i')
    ban_list = eval(ban)
    ban_dict = {}
    for num in ban_list:
        ban_dict[num] = 1

    tot_instance = 0
    for file in lp_files:
        #读入训练文件
        if((int)(file[pre_len:-3]) in ban_dict.keys()):
            print("setcovers:", (int)(file[pre_len:-3]))
            continue
        if(tot_instance >= max_num):
            break
        now_num = file[pre_len:-3]
        now_lp_file = file
        now_pickle_file = '/home/sharing/disk1/yehuigen/_latest_datasets/MIPcc23-Dataset/vary_rhs/series_3/Pickle/rhs_s3_i' + now_num + '.pickle'

        if(os.path.exists(now_lp_file) == False):
            print("No LP file!")
        if(os.path.exists(now_pickle_file) == False):
            print("No Pickle file!")
        
        #创建模型和最优解读入
        model = read(now_lp_file)
        with open(now_pickle_file, "rb") as f:
            data = pickle.load(f)
        solution = data[0]
        gap = data[1]


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
            now_solution[value_to_num[val.VarName]] = solution[val.VarName]

        #1最小化，-1最大化
        obj_type = model.ModelSense


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
        
        #图划分并打包
        partition_num = 10
        vertex_num = n + m
        edge_num = 0

        edge = []
        for i in range(vertex_num):
            edge.append([])
        for i in range(m):
            for j in range(k[i]):
                edge[site[i][j]].append(n + i)
                edge[n + i].append(site[i][j])
                edge_num += 2
        
        alpha = (partition_num ** 0.5) * edge_num / (vertex_num ** (2 / 3))
        gamma = 1.5
        balance = 1.1
        #print(alpha)

        visit = np.zeros(vertex_num, int)
        order = []
        for i in range(vertex_num):
            if(visit[i] == 0):
                q = []
                q.append(i)
                visit[q] = 1
                now = 0
                while(now < len(q)):
                    order.append(q[now])
                    for neighbor in edge[q[now]]:
                        if(visit[neighbor] == 0):
                            q.append(neighbor)
                            visit[neighbor] = 1
                    now += 1
        

        #print(len(order))
        color = np.zeros(vertex_num, int)
        for i in range(vertex_num):
            color[i] = -1
        cluster_num = np.zeros(partition_num)

        for i in range(vertex_num):
            now_vertex = order[i]
            load_limit = balance * vertex_num / partition_num
            score = np.zeros(partition_num, float)
            for neighbor in edge[now_vertex]:
                if(color[neighbor] != -1):
                    score[color[neighbor]] += 1
            for j in range(partition_num):
                if(cluster_num[j] < load_limit):
                    score[j] -= alpha * gamma * (cluster_num[j] ** (gamma - 1))
                else:
                    score[j] = -1e9
            
            now_score = -2e9
            now_site = -1
            for j in range(partition_num):
                if(score[j] > now_score):
                    now_score = score[j]
                    now_site = j
            
            color[now_vertex] = now_site
            cluster_num[now_site] += 1

        #print(cluster_num)

        new_color = []
        for i in range(n):
            new_color.append(color[i])
        
        folder_path = './example_rhs_series_3_cc'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open('./example_rhs_series_3_cc' + '/pair' + str(tot_instance) + '.pickle', 'wb') as f:
            pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, now_solution], f)
        
        tot_instance += 1



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type = str, default = 'IS', help = 'The number of instances.')
    parser.add_argument("--difficulty", type = str, default = 'easy', help = 'The number of instances.')
    parser.add_argument("--ban", type = str, default = '[5,7,9,14,21]', help = 'The number of instances.')
    parser.add_argument("--max_num", type = int, default = 50, help = 'The number of instances.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    #print(vars(args))
    generate_pair(**vars(args))