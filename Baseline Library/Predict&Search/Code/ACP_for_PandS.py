import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import random
import pickle
import time
import os

def find_all(root_folder, string= 'LP'):
    '''
    Return the folder paths of all files containing 'LP' under root_folder and store them in a list
    '''
    file_paths = []
    for root, dir, files in os.walk(root_folder):
        if string in dir:
            file_paths.append(root)
    return(file_paths)

def split_problem(lp_file):
    '''
    Pass in the lp file and split the incoming questions
    '''
    model = gp.read(lp_file)
    value_to_num = {}
    value_num = 0
    #N represents the number of decision variables 
    #M represents the number of constraints 
    #K [i] represents the number of decision variables in the i-th constraint 
    #Site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint 
    #Value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint 
    #Constraint [i] represents the number to the right of the i-th constraint 
    #Constrict_type [i] represents the type of the i-th constraint, 1 represents<, 2 represents>, and 3 represents= 
    #Coefficient [i] represents the coefficient of the i-th decision variable in the objective function
    n = model.NumVars
    m = model.NumConstrs
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

    for val in model.getVars():
        if(val.VarName not in value_to_num.keys()):
            value_to_num[val.VarName] = value_num
            value_num += 1
        coefficient[value_to_num[val.VarName]] = val.Obj
        lower_bound[value_to_num[val.VarName]] = val.LB
        upper_bound[value_to_num[val.VarName]] = val.UB
        value_type[value_to_num[val.VarName]] = val.Vtype

    #1 minimize，-1 maximize
    obj_type = model.ModelSense
    if(obj_type == -1):
        obj_type = 'maximize'
    else:
        obj_type = 'minimize'
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num

def get_ans(lp_file, value_to_num):
    lp_name = os.path.split(lp_file)[1]
    problem_name = os.path.split(os.path.split(lp_file)[0])[1]
    pickle_file = f'instance/pickles/{problem_name}/{lp_name[:-3]}.pickle'
    with open(pickle_file, 'rb')as f:
        x_dict, gap = pickle.load(f)
    model = gp.read(lp_file)
    obj_expr = model.getObjective()
    obj_val = 0
    for i in range(obj_expr.size()):
        obj_val += x_dict[obj_expr.getVar(i).VarName] * obj_expr.getCoeff(i)
    obj_val += obj_expr.getConstant()
    print(f'obj_val: {obj_val}')

    ansx = np.zeros(model.NumVars)
    for var in model.getVars():
        ansx[value_to_num[var.VarName]] = x_dict[var.VarName]
    return obj_val, ansx

# def get_ans(lp_file, value_to_num):
    model = gp.read(lp_file)
    model.Params.PoolSolutions = 1
    model.Params.PoolSearchMode = 1
    model.setParam('TimeLimit', 100)
    model.setParam('MIPGap', 20)
    model.optimize()
    ansx = np.zeros(model.NumVars)
    # 输出初始解
    if model.SolCount > 0:
        ans = model.ObjVal

        for var in model.getVars():
            ansx[value_to_num[var.VarName]] = var.X
    else:
        print("No initial solution found.")
    
    return ans, ansx


def run_ACP(lp_file, settings, block = 2, time_limit = 100, max_turn_ratio = 0.1):
    '''
    Function Description:
    Based on the provided problem instance, use the Gurobi solver as a sub solver for ACP to solve.

    Parameter description:
    -lp_file: The path of lp file
    -settings: Settings dict
    -block: Initial blocking num
    -time_limit: Time limit for this problem
    -max_turn_ratio: The maximum proportion of ACP algorithm running in each round to the total time limit
    '''
    #Set KK as the initial number of blocks and PP as the selected number of blocks, that is, constrain the selection of PP blocks after dividing into KK blocks for optimization
    KK = block
    PP = 1
    max_turn = 5
    epsilon = 0.01
    #Obtain the split problem model
    max_turn_time = max_turn_ratio * time_limit
    n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem(lp_file)    
    #Establish a variable dictionary
    num_to_value = {value : key for key, value in value_to_num.items()}

    #begin time
    begin_time = time.time()
    
    #Initialize initial solution and initial answer
    ans, ansx = get_ans(lp_file, value_to_num)
    print(f"初始解目标值为：{ans}")
    
    #The partition label of the constraint, cons_comor [i] represents which block the i-th constraint belongs to
    cons_color = np.zeros(m, int)
    last = ans

    sols = []
    objs = []
    turn = 1

    #Obtain the variable dictionary of the original problem
    ori_model = gp.read(lp_file)
    oriVarNames = [var.varName for var in ori_model.getVars()]

    while(time.time() - begin_time <= time_limit and len(objs) < settings['maxsol']):
        print("KK = ", KK)
        print("PP = ", PP)
        #Randomly divide constraints into KK blocks
        for i in range(m):
            cons_color[i] = random.randint(1, KK)
        now_cons_color = 1
        #Set all decision variables involved in a random constraint as the variables to be optimized
        #Color [i]=1 indicates that the i-th decision variable is selected as the variable to be optimized
        color = np.zeros(n, int)
        now_color = 1
        color_num = 0
        for i in range(m):
            if(PP == 1 and cons_color[i] == now_cons_color):
                for j in range(k[i]):
                    color[site[i][j]] = 1
                    color_num += 1
            if(PP > 1 and cons_color[i] != now_cons_color):
                for j in range(k[i]):
                    color[site[i][j]] = 1
                    color_num += 1
        #Site_to-color [i] represents who is the i-th decision variable in the variables to be optimized
        #Color_to-site [i] represents the i-th decision variable mapped to the i-th decision variable in the variable to be optimized
        #Vertex_color_num represents the number of decision variables in the variables to be optimized
        site_to_color = np.zeros(n, int)
        color_to_site = np.zeros(n, int)
        vertex_color_num = 0
        #Define model
        model = gp.Model("ACP")
        x = []
        for i in range(n):
            if(color[i] == now_color):
                color_to_site[vertex_color_num] = i
                site_to_color[i] = vertex_color_num
                vertex_color_num += 1
                if(value_type[i] == 'B'):
                    now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY)
                elif(value_type[i] == 'I'):
                    now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER)
                else:
                    now_val = model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS)
                x.append(now_val)
        #Define n decision variables x []
        #X=model. addMVar (vertex_color_num), lb=0, ub=1, vtype=GRB. BINARY) # lb variable lower limit, ub variable upper limit
        #Set objective function and optimization objective (maximize/minimize), only optimize the variables to be optimized
        objsum = 0
        objtemp = 0
        for i in range(n):
            if(color[i] == now_color):
                objsum += x[site_to_color[i]] * coefficient[i]
            else:
                objtemp += ansx[i] * coefficient[i]
        if(obj_type == 'maximize'):
            model.setObjective(objsum, GRB.MAXIMIZE)
        else:
            model.setObjective(objsum, GRB.MINIMIZE)
        #Add m constraints, only add the constraints from the randomly selected one
        for i in range(m): 
            flag = 0
            constr = 0
            for j in range(k[i]):
                if(color[site[i][j]] == now_color):
                    flag = 1
                    constr += x[site_to_color[site[i][j]]] * value[i][j]
                else:
                    constr += ansx[site[i][j]] * value[i][j]
            if(flag):
                if(constraint_type[i] == 1):
                    model.addConstr(constr <= constraint[i])
                elif(constraint_type[i] == 2):
                    model.addConstr(constr >= constraint[i])
                else:
                    model.addConstr(constr == constraint[i])
        

        model.setParam('TimeLimit', min(max(time_limit - (time.time() - begin_time), 0), max_turn_time))
        model.optimize()
        
        try:
            #Calculate the current target value
            temp = model.ObjVal + objtemp
            print(f"当前目标值为：{temp}")
            bestX = []
            for i in range(vertex_color_num):
                bestX.append(x[i].X)
            #print(bestX)

            if(obj_type == 'maximize'):
                #Update the current optimal solution and optimal answer
                if(temp > ans):
                    for i in range(vertex_color_num):
                        ansx[color_to_site[i]] = bestX[i]
                    ans = temp
                #Adaptive block number variation
                if((ans - last <= epsilon * ans)):
                    turn += 1
                    if(turn == max_turn):
                        if(KK > 2 and PP == 1):
                            KK -= 1
                        else:
                            KK += 1
                            PP += 1
                        turn = 1
                else:
                    turn = 0
            else:
                if(temp < ans):
                    for i in range(vertex_color_num):
                        ansx[color_to_site[i]] = bestX[i]
                    ans = temp
                #Adaptive block number variation
                if((last - ans <= epsilon * ans)):
                    turn += 1
                    if(turn == max_turn):
                        if(KK > 2 and PP == 1):
                            KK -= 1
                        else:
                            KK += 1
                            PP += 1
                        turn = 1
                else:
                    turn = 0
            
            if(model.MIPGap != 0):
                if(KK == 2 and PP > 1):
                    KK -= 1
                    PP -= 1
                else:
                    KK += 1
                turn = 0
            last = ans
        except:
            try:
                model.computeIIS()
                if(KK > 2 and PP == 1):
                    KK -= 1
                else:
                    KK += 1
                    PP += 1
                turn = 1
            except:
                if(KK == 2 and PP > 1):
                    KK -= 1
                    PP -= 1
                else:
                    KK += 1
                turn = 0                    
            print("This turn can't improve more")


        sols.append(ansx)
        objs.append(ans)

    # oriVarNames = [var.varname for var in model.getVars()]
    sol_data = {
        'var_names': oriVarNames,
        'sols': sols,
        'objs': objs,
    }
    return sol_data
