# This script generates random MILP instances for training and testing

import numpy as np
import random as rd
import os
import pickle
import argparse
import gurobipy as gp
from gurobipy import GRB
from pandas import read_csv 
import pyscipopt as scip
import json


def generate_one(lp_file, pickle_file ,path):

    if not os.path.exists(path):
        os.makedirs(path)
    model = gp.read(lp_file)
    n = model.NumVars
    m = model.NumConstrs
    constraint = []
    circ = []
    name_to_num = {}
    num_to_name = {}
    lb = []
    ub = []
    NI = []#According to the variable index, the type of the variable is 1 for a shaped variable and 0 for a continuous variable

    i = 0
    for var in model.getVars():
        name_to_num[var.VarName] = i
        num_to_name[i] = var.VarName
        i+=1

        if var.VType == GRB.CONTINUOUS:
            NI.append(0)
            lb.append(var.LB)
            ub.append(var.UB)
        elif var.VType == GRB.INTEGER:
            NI.append(1)
            lb.append(var.LB)
            ub.append(var.UB)
        else: #binary
            NI.append(1)
            lb.append(0)
            ub.append(1)

    for cnstr in model.getConstrs():
        if(cnstr.Sense == '<'):
            circ.append(0)
        elif(cnstr.Sense == '='):
            circ.append(1) 
        else:
            circ.append(2) 
        constraint.append(cnstr.RHS)

    var_num = len(name_to_num)
    
    b = np.array(constraint).reshape(m, 1)
    circ = np.array(circ).reshape(m, 1)

    def get_coefficient(expr, var):
        """Retrieve the coefficient of the specified variable in the expression, and return 0 if the variable does not exist"""
        for i in range(expr.size()):
            if expr.getVar(i).VarName == var.VarName:
                return expr.getCoeff(i)
        return 0  
    
    EdgeFeature = []
    EdgeIndex = []
    now_cnstr = 0
    for cnstr in model.getConstrs():
        row = model.getRow(cnstr)
        tmp_list = []
        for i in range(row.size()):
            EdgeIndex.append([n + now_cnstr,
                                name_to_num[row.getVar(i).VarName]])
            EdgeFeature.append(row.getCoeff(i))
        now_cnstr += 1       
        

    EdgeIndex = np.array(EdgeIndex)

    #c, NI, lb, ub
    obj_expr = model.getObjective()
    c = []
    # 修改
    for i in range(var_num):
        tmp_var = model.getVarByName(num_to_name[i])
        c.append(get_coefficient(obj_expr, tmp_var))
    c = np.array(c)
    # print(f'm: {m}')
    # print(f'n(varnums): {n}')
    # print(f'b: {b.shape}')
    # print(f'var_num: {var_num}')
    # print(f'EdgeIndex: {EdgeIndex.shape}')
    # print(f'EdgeFeature: {np.array(EdgeFeature).shape}')
    # print(f'c: {c.shape}')


    np.savetxt(f'{path}/ConFeatures.csv', np.hstack((b, circ)), delimiter = ',', fmt = '%10.5f')
    np.savetxt(f'{path}/EdgeFeatures.csv', EdgeFeature, fmt = '%10.5f')
    np.savetxt(f'{path}/EdgeIndices.csv', EdgeIndex, delimiter = ',', fmt = '%d')
    np.savetxt(f'{path}/VarFeatures.csv', np.hstack((c.reshape(n, 1), np.array(NI).reshape(n, 1), np.array(lb).reshape(n, 1), np.array(ub).reshape(n, 1))), delimiter = ',', fmt = '%10.5f')

    #加载解
    with open(pickle_file, 'rb')as f:
        x_dict, gap = pickle.load(f)

    sol = []
    for i in range(var_num):
        sol.append(x_dict[num_to_name[i]])
    sol = np.array(sol).T
    obj = 0
    for i in range(obj_expr.size()):
        obj += x_dict[obj_expr.getVar(i).VarName] * obj_expr.getCoeff(i)
    np.savetxt(f'{path}/Labels_feas.csv', [1], fmt = '%d')
    np.savetxt(f'{path}/Labels_obj.csv', [obj], fmt = '%10.5f')
    np.savetxt(f'{path}/Labels_solu.csv', sol, fmt = '%10.5f')
    
    #1最小化，-1最大化
    if(model.ModelSense == 1):
        obj_type = 'minimize'
    else:
        obj_type = 'maximize'
    return var_num, m, len(EdgeIndex), obj_type

def convert_to_float(array):
    for i, row in enumerate(array):
        for j, val in enumerate(row):
            try:
                array[i][j] = float(val)
            except ValueError:
                if array[i][j] ==  'inf':  # Or appropriately handle non numeric values
                    array[i][j] = np.inf
                else:
                    array[i][j] = np.nan
    return array

## DATA GENERATION

def generate_all(lp_folder:str, data_path):
    lp_names = sorted(f for f in os.listdir(lp_folder)if f.endswith('.lp'))
    json_dict = {}
    for lp_name in lp_names:
        lp_file = os.path.join(lp_folder, lp_name)
        instance_data_path = f'{data_path}/{lp_name}_data'
        pickle_file = f'{os.path.split(lp_folder)[0]}/Pickle/{lp_name[:-3]}.pickle'
        var_num, cons_num, nonzero_num, obj_type= generate_one(lp_file, pickle_file, instance_data_path)
        json_dict[f'{lp_name}_cons'] = cons_num
        json_dict[f'{lp_name}_vars'] = var_num
        json_dict[f'{lp_name}_nonzero'] = nonzero_num
        json_dict[f'{lp_name}_obj'] = obj_type
    with open(f'{data_path}/cons_vars_dict.json', 'w')as f:
        json.dump(json_dict, f)
      


def combineGraphsAll(lp_folder, load_data_path, save_data_path, is_append_rand):

    lp_names = sorted(f for f in os.listdir(lp_folder)if f.endswith('.lp'))
    with open(f'{load_data_path}/cons_vars_dict.json', 'r')as f:
        json_dict = json.load(f)

    def my_stack(ori,aug):
        return (np.copy(aug) if ori is None else np.concatenate((ori,aug),axis=0) )

    startMIPidx_m = 0
    startMIPidx_n = 0
    last_m = -1
    last_n = -1
    m_all = []
    n_all = []
    nonzero_all = []
    obj_type_all = []
    same_flag = 1
    ConFeatures_all = None
    EdgeFeatures_all = None
    EdgeIndices_all = None
    VarFeatures_all = None
    Labels_feas = None

    for lp_name in (lp_names):

        MIP_path = f"{load_data_path}/{lp_name}_data"
        
        varFeatures = read_csv(f"{MIP_path}/VarFeatures.csv", header=None).values
        conFeatures = read_csv(f"{MIP_path}/ConFeatures.csv", header=None).values
        edgeFeatures = read_csv(f"{MIP_path}/EdgeFeatures.csv", header=None).values
        edgeIndices = read_csv(f"{MIP_path}/EdgeIndices.csv", header=None).values
        labelsFeas = read_csv(f"{MIP_path}/Labels_feas.csv", header=None).values
        m = json_dict[f'{lp_name}_cons']
        n = json_dict[f'{lp_name}_vars']
        nonzero = json_dict[f'{lp_name}_nonzero']
        obj_type = json_dict[f'{lp_name}_obj']
        m_all.append(m)
        n_all.append(n)
        nonzero_all.append(nonzero)
        obj_type_all.append(obj_type)
        edgeIndices[:, 0] = edgeIndices[:, 0] #+ startMIPidx_m
        edgeIndices[:, 1] = edgeIndices[:, 1] #+ startMIPidx_n

        ConFeatures_all = my_stack(ConFeatures_all, conFeatures)
        VarFeatures_all = my_stack(VarFeatures_all, varFeatures)
        EdgeFeatures_all = my_stack(EdgeFeatures_all, edgeFeatures)
        EdgeIndices_all = my_stack(EdgeIndices_all, edgeIndices)
        Labels_feas = my_stack(Labels_feas, labelsFeas)

        startMIPidx_m += m
        startMIPidx_n += n
        if(last_m > 0 and m != last_m):
            same_flag = 0
        if(last_n > 0 and n != last_n):
            same_flag = 0
        last_m = m
        last_n = n

    if is_append_rand:
        print('Before appending:',ConFeatures_all.shape,VarFeatures_all.shape)
        if(same_flag):
            kkk = ConFeatures_all.shape[0] // m
            np.random.seed(0)
            ConAug = np.tile(np.random.rand(m,1), (kkk,1))
            VarAug = np.tile(np.random.rand(n,1), (kkk,1))
        else:
            np.random.seed(0)
            ConAug = np.tile(np.random.rand(startMIPidx_m,1), (1,1))
            VarAug = np.tile(np.random.rand(startMIPidx_n,1), (1,1))
        ConFeatures_all = np.concatenate((ConFeatures_all, ConAug),axis=1)
        VarFeatures_all = np.concatenate((VarFeatures_all, VarAug),axis=1)
        print('After appending:',ConFeatures_all.shape,VarFeatures_all.shape)

    ConFeatures_all = convert_to_float(ConFeatures_all)
    EdgeFeatures_all = convert_to_float(EdgeFeatures_all)
    EdgeIndices_all = convert_to_float(EdgeIndices_all)
    VarFeatures_all = convert_to_float(VarFeatures_all)
    Labels_feas = convert_to_float(Labels_feas)
    # print(f'\nEdgeFeatures_all: {EdgeFeatures_all.dtype}\n')


    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    np.savetxt(f"{save_data_path}/ConFeatures_all.csv", ConFeatures_all, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/EdgeFeatures_all.csv", EdgeFeatures_all, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/EdgeIndices_all.csv", EdgeIndices_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/VarFeatures_all.csv", VarFeatures_all, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/Labels_feas.csv", Labels_feas, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/Con_num.csv", m_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/Var_num.csv", n_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/Nonzero_num.csv", nonzero_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/Obj_type.csv", obj_type_all, delimiter = ',', fmt = '%s')
    

def combineGraphsFeas(lp_folder, load_data_path, save_data_path, is_append_rand):
    '''
    This function combines all feasible MILP instances in "folder".
    This function also makes labels for the optimal objective and optimal solution.
    '''

    lp_names = sorted(f for f in os.listdir(lp_folder)if f.endswith('.lp'))
    with open(f'{load_data_path}/cons_vars_dict.json', 'r')as f:
        json_dict = json.load(f)

    def my_stack(ori,aug):
        return (np.copy(aug) if ori is None else np.concatenate((ori,aug),axis=0) )

    ConFeatures_feas = None
    EdgeFeatures_feas = None
    EdgeIndices_feas = None
    VarFeatures_feas = None
    Labels_solu = None
    Labels_obj = None
    startMIPidx_m = 0
    startMIPidx_n = 0
    last_m = -1
    last_n = -1
    same_flag = 1
    m_all = []
    n_all = []
    nonzero_all = []
    obj_type_all = []

    for lp_name in lp_names:

        MIP_path = f"{load_data_path}/{lp_name}_data"
        varFeatures = read_csv(f"{MIP_path}/VarFeatures.csv", header=None).values
        conFeatures = read_csv(f"{MIP_path}/ConFeatures.csv", header=None).values
        edgeFeatures = read_csv(f"{MIP_path}/EdgeFeatures.csv", header=None).values
        edgeIndices = read_csv(f"{MIP_path}/EdgeIndices.csv", header=None).values
        labelsObj = read_csv(f"{MIP_path}/Labels_obj.csv", header=None).values
        labelsSolu = read_csv(f"{MIP_path}/Labels_solu.csv", header=None).values
        
        m = json_dict[f'{lp_name}_cons']
        n = json_dict[f'{lp_name}_vars']
        nonzero = json_dict[f'{lp_name}_nonzero']
        obj_type = json_dict[f'{lp_name}_obj']
        m_all.append(m)
        n_all.append(n)
        nonzero_all.append(nonzero)
        obj_type_all.append(obj_type)

        edgeIndices[:, 0] = edgeIndices[:, 0] #+ startMIPidx_m
        edgeIndices[:, 1] = edgeIndices[:, 1] #+ startMIPidx_n
        
        ConFeatures_feas = my_stack(ConFeatures_feas, conFeatures)
        VarFeatures_feas = my_stack(VarFeatures_feas, varFeatures)
        EdgeFeatures_feas = my_stack(EdgeFeatures_feas, edgeFeatures)
        EdgeIndices_feas = my_stack(EdgeIndices_feas, edgeIndices)
        Labels_solu = my_stack(Labels_solu, labelsSolu)
        Labels_obj = my_stack(Labels_obj, labelsObj)

        startMIPidx_m += m
        startMIPidx_n += n
        if(last_m > 0 and m != last_m):
            same_flag = 0
        if(last_n > 0 and n != last_n):
            same_flag = 0
        last_m = m
        last_n = n

    if is_append_rand:
        print('Before appending:',ConFeatures_feas.shape,VarFeatures_feas.shape)
        if(same_flag):
            kkk = ConFeatures_feas.shape[0] // m
            np.random.seed(0)
            ConAug = np.tile(np.random.rand(m,1), (kkk,1))
            VarAug = np.tile(np.random.rand(n,1), (kkk,1))
        else:
            np.random.seed(0)
            ConAug = np.tile(np.random.rand(startMIPidx_m,1), (1,1))
            VarAug = np.tile(np.random.rand(startMIPidx_n,1), (1,1))
        ConFeatures_feas = np.concatenate((ConFeatures_feas, ConAug),axis=1)
        VarFeatures_feas = np.concatenate((VarFeatures_feas, VarAug),axis=1)
        print('After appending:',ConFeatures_feas.shape,VarFeatures_feas.shape)
        
    ConFeatures_feas = convert_to_float(ConFeatures_feas)
    EdgeFeatures_feas = convert_to_float(EdgeFeatures_feas)
    EdgeIndices_feas = convert_to_float(EdgeIndices_feas)
    VarFeatures_feas = convert_to_float(VarFeatures_feas)
    Labels_solu = convert_to_float(Labels_solu)
    Labels_obj = convert_to_float(Labels_obj)

    if not os.path.exists(save_data_path):
        os.mkdir(save_data_path)
    np.savetxt(f"{save_data_path}/ConFeatures_feas.csv", ConFeatures_feas, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/EdgeFeatures_feas.csv", EdgeFeatures_feas, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/EdgeIndices_feas.csv", EdgeIndices_feas, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/VarFeatures_feas.csv", VarFeatures_feas, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/Labels_solu.csv", Labels_solu, delimiter = ',', fmt = '%10.5f')
    np.savetxt(f"{save_data_path}/Labels_obj.csv", Labels_obj, delimiter = ',', fmt = '%10.5f')	
    np.savetxt(f"{save_data_path}/Con_num.csv", m_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/Var_num.csv", n_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/Nonzero_num.csv", nonzero_all, delimiter = ',', fmt = '%d')
    np.savetxt(f"{save_data_path}/Obj_type.csv", obj_type_all, delimiter = ',', fmt = '%s')



rd.seed(0)


## MAIN SCRIPT
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        choices=['setcover', 'cauctions', 'facilities', 'indset',
                 '1_item_placement',
                 'Cut',
                 'knapsack',
                 'MIPlib',
                 'mis',
                 'Nexp',
                 'setcover',
                 'Transportation',
                 'vary_bounds_s1',
                 'vary_matrix_rhs_bounds_obj_s1',
                 'vary_matrix_s1',
                 'vary_obj_s1',
                 'vary_obj_s3',
                 'vary_rhs_obj_s2',
                 'vary_rhs_s2',
                 'vary_rhs_s4',],
    )
    args = parser.parse_args()
    problem_name = args.problem

    problem_folder = f'instances/{problem_name}'

    lp_folder_train = f'{problem_folder}/LP'
    lp_folder_test = f'{problem_folder}/LP_test'

    train_data_path = f'{problem_folder}/train_csv'
    test_data_path = f'{problem_folder}/test_csv'

    save_train_data_path = f'{problem_folder}/train'
    save_test_data_path = f'{problem_folder}/test'

    for fol in [lp_folder_train, lp_folder_test, train_data_path, test_data_path, save_train_data_path, save_test_data_path]:
        if not os.path.exists(fol):
            os.makedirs(fol)

    generate_all(lp_folder_train, train_data_path)
    generate_all(lp_folder_test, test_data_path)

    combineGraphsAll(lp_folder_train, train_data_path, save_train_data_path, True)
    combineGraphsFeas(lp_folder_train, train_data_path, save_train_data_path, True)

    combineGraphsAll(lp_folder_test, test_data_path, save_test_data_path, True)
    combineGraphsFeas(lp_folder_test, test_data_path, save_test_data_path, True)