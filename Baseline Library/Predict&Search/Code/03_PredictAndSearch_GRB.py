import pandas as pd
import gurobipy
from gurobipy import GRB
import argparse
import random
import os
import numpy as np
import torch
from helper import get_a_new2
import time
import pickle

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

#4 public datasets, IS, WA, CA, IP

# def test_hyperparam(task):
#     '''
#     set the hyperparams
#     k_0, k_1, delta
#     '''
#     if task=="IP":
#         return 400,5,1
#     elif task == "IS":
#         return 300,300,15
#     elif task == "WA":
#         return 0,600,5
#     elif task == "CA":
#         return 400,0,10
#     else:
#         return 0,0,18
# k_0,k_1,delta=test_hyperparam(TaskName)

#set log folder
solver='GRB'

DEVICE = torch.device("cuda:1")
DEVICE = torch.device("cpu")
# Aclib, CORAL, 'Cut', 'MIPlib', 'vary_matrix_rhs_bounds_obj_s1', 'vary_matrix_rhs_bounds_s1', 'vary_matrix_s1', 'vary_obj_s1', 'vary_obj_s2', 'vary_obj_s3', 'vary_rhs_obj_s1', 'vary_rhs_obj_s2', 'vary_rhs_s1', 'vary_rhs_s2', 'vary_rhs_s3', 'vary_rhs_s4', easy_knaspack, easy_mis, easy_setcover, fc.data, medium_corlat, medium_mik, nn_verification

test_params = [ ['vary_bounds_s1', 400],
                ['vary_bounds_s2', 1000],
                ['vary_bounds_s3', 1000],
                ['vary_matrix_s1', 100],
                ['vary_matrix_rhs_bounds_obj_s1', 100],
                ['vary_matrix_rhs_bounds_s1', 100],
                ['vary_obj_s1', 100],
                ['vary_obj_s2', 150],
                ['vary_obj_s3', 100],
                ['vary_rhs_s1', 100],
                ['vary_rhs_s2', 100],
                ['vary_rhs_s3', 100],
                ['vary_rhs_s4', 100],
                ['vary_rhs_obj_s1', 600],
                ['vary_rhs_obj_s2', 100],
                ['Transportation', 4000],
                ['MIPlib', 150],
                ['3_anonymous', 4000],
                ['CORAL', 4000],
                ['ECOGCNN', 4000],
                ['Nexp', 4000],
                ['Cut', 4000],
                ['IS_easy_instance', 600],
                ['SC_easy_instance', 600],
                ['MVC_easy_instance', 600],
                ['IS_medium_instance', 2000],
                ['IS_medium_instance', 2000],
                ['IS_medium_instance', 2000],
                ['Aclib', 100],
                ['fc.data', 100],
                ['nn_verification', 100],
                ['easy_knaspack', 100],
                ['easy_mis', 100],
                ['easy_setcover', 100],
                ['medium_corlat', 100],
                ['medium_mik', 100],
                ]

for params in test_params:

    TaskName = params[0]
    time_limit = params[1]


    test_task = f'{TaskName}_{solver}_Predect&Search'
    if not os.path.isdir(f'./logs'):
        os.mkdir(f'./logs')
    if not os.path.isdir(f'./logs/{TaskName}'):
        os.mkdir(f'./logs/{TaskName}')
    if not os.path.isdir(f'./logs/{TaskName}/{test_task}'):
        os.mkdir(f'./logs/{TaskName}/{test_task}')
    log_folder=f'./logs/{TaskName}/{test_task}'


    #load pretrained model
    if TaskName=="IP":
        #Add position embedding for IP model, due to the strong symmetry
        from GCN import GNNPolicy_position as GNNPolicy,postion_get
    else:
        from GCN import GNNPolicy
    pathstr=f'pretrain/{TaskName}_train/model_last.pth'
    policy = GNNPolicy().to(DEVICE)
    state = torch.load(pathstr, map_location=torch.device('cuda:1'))
    policy.load_state_dict(state)


    sample_names = sorted(file for file in os.listdir(f'instance/test/{TaskName}') if file.endswith('.lp'))

    results = []
    for lp_name in sample_names:
        ins_name_to_read = f'instance/test/{TaskName}/{lp_name}'

        #get bipartite graph as input
        A, v_map, v_nodes, c_nodes, b_vars=get_a_new2(ins_name_to_read)
        constraint_features = c_nodes.cpu()
        constraint_features[np.isnan(constraint_features)] = 1 #remove nan value
        variable_features = v_nodes
        if TaskName == "IP":
            variable_features = postion_get(variable_features)
        edge_indices = A._indices()
        edge_features = A._values().unsqueeze(1)
        edge_features=torch.ones(edge_features.shape)

        #prediction
        BD = policy(
            constraint_features.to(DEVICE),
            edge_indices.to(DEVICE),
            edge_features.to(DEVICE),
            variable_features.to(DEVICE),
        ).sigmoid().cpu().squeeze()

        #align the variable name betweend the output and the solver
        all_varname=[]
        for name in v_map:
            all_varname.append(name)
        binary_name=[all_varname[i] for i in b_vars]
        scores=[]#get a list of (index, VariableName, Prob, -1, type)
        for i in range(len(v_map)):
            type="C"
            if all_varname[i] in binary_name:
                type='BINARY'
            scores.append([i, all_varname[i], BD[i].item(), -1, type])


        scores.sort(key=lambda x:x[2],reverse=True)

        scores=[x for x in scores if x[4]=='BINARY']#get binary

        k_0, k_1 = 0, 0

        fixer=0
        #fixing variable picked by confidence scores
        count1=0
        for i in range(len(scores)):
            if count1<k_1:
                scores[i][3] = 1
                count1+=1
                fixer += 1
        scores.sort(key=lambda x: x[2], reverse=False)
        count0 = 0
        for i in range(len(scores)):
            if count0 < k_0:
                scores[i][3] = 0
                count0 += 1
                fixer += 1


        print(f'instance: {lp_name}, ')


        #read instance
        gurobipy.setParam('LogToConsole', 1)  # hideout
        m = gurobipy.read(ins_name_to_read)
        m.Params.TimeLimit = time_limit
        m.Params.Threads = 1
        m.Params.MIPFocus = 1
        m.Params.LogFile = f'{log_folder}/{lp_name}.log'
        delta = 0.1*m.NumConstrs
        # trust region method implemented by adding constraints
        instance_variabels = m.getVars()
        instance_variabels.sort(key=lambda v: v.VarName)
        variabels_map = {}
        for v in instance_variabels:  # get a dict (variable map), varname:var clasee
            variabels_map[v.VarName] = v
        alphas = []
        for i in range(len(scores)):
            tar_var = variabels_map[scores[i][1]]  # target variable <-- variable map
            x_star = scores[i][3]  # 1,0,-1, decide whether need to fix
            if x_star < 0:
                continue
            # tmp_var = m1.addVar(f'alp_{tar_var}', 'C')
            tmp_var = m.addVar(name=f'alp_{tar_var}', vtype=GRB.CONTINUOUS)
            alphas.append(tmp_var)
            m.addConstr(tmp_var >= tar_var - x_star, name=f'alpha_up_{i}')
            m.addConstr(tmp_var >= x_star - tar_var, name=f'alpha_dowm_{i}')
        all_tmp = 0
        for tmp in alphas:
            all_tmp += tmp
        m.addConstr(all_tmp <= delta, name="sum_alpha")
        start_time = time.time()
        m.optimize()
        results.append({'Filename': lp_name,
                        'MIPGap': m.MIPGap,
                        'Solve_time': time.time()-start_time})
        df = pd.DataFrame(results)
        # 将{文件名，gap，求解时间}储存在file_path下的A_GUROBI_output.xlsx中
        df.to_excel(f'instance/test/{TaskName}/A_P&S_output.xlsx',index = False)
        print(f'Done excel!')

        pickle_path = f'instance/test/{TaskName}/P&S_Pickle'
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        ans = {}
        try:
            for val in m.getVars():
                ans[val.VarName] = val.x
            with open(pickle_path + '/' + lp_name[:-3] + '.pickle', 'wb') as f:
                pickle.dump([ans, m.MIPGap], f)
        except:
            for val in m.getVars():
                ans[val.VarName] = val.x
            with open(pickle_path + '/' + lp_name[:-3] + '.pickle', 'wb') as f:
                pickle.dump([ans, None], f)







