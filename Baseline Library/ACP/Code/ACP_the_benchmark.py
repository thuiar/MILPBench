import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import random
import pickle
import time
import os
from alive_progress import alive_bar

def find_all(root_folder, string= 'LP'):
    '''
    Return a list of folders containing the string 'LP' in their names, found under the 'root_folder'
    '''
    file_paths = []
    for root, dir, files in os.walk(root_folder):
        if string in dir:
            file_paths.append(root)
    return(file_paths)

def split_problem(lp_file):
    '''
    Pass in an LP file and split the given problem
    '''
    model = gp.read(lp_file)
    value_to_num = {}
    value_num = 0
    #n represents the num of decision variables
    #m represents the num of constrains
    #k[i] represents the number of decision variables in the i-th constraint
    #site[i][j] represents which decision variable is the j-th decision variable in the i-th constraint
    #value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint
    #constraint[i] represents the number on the right-hand side of the i-th constraint
    #constraint_type[i] represents the type of the i-th constraint, where 1 indicates '<', 2 indicates '>', and 3 indicates '='
    #coefficient[i] represents the coefficient of the i-th decision variable in the objective function
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

    #1 for maximize，-1 for minimize
    obj_type = model.ModelSense
    if(obj_type == -1):
        obj_type = 'maximize'
    else:
        obj_type = 'minimize'
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num

def get_ans(lp_file, time_limit, value_to_num):
    model = gp.read(lp_file)
    model.Params.PoolSolutions = 1
    model.Params.PoolSearchMode = 1
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 20)
    model.optimize()
    ansx = np.zeros(model.NumVars)
    if model.SolCount > 0:
        ans = model.ObjVal

        for var in model.getVars():
            ansx[value_to_num[var.VarName]] = var.X
    else:
        print("No initial solution found.")
    
    return ans, ansx

def run_ACP(lp_file, block = 2, time_limit = 4000, max_turn_ratio = 0.1):
    '''
    Run LNS (Large Neighborhood Search), passing in the lp file. 'block' is the number of blocks (default 2), 'time_limit' is the total running time limit (default 4000), and 'max_turn_ratio' is the maximum running time ratio for each turn (default 0.1).
    '''
    #Set KK as the initial number of blocks and PP as the selected number of blocks to optimize after dividing the constraints into KK blocks
    KK = block
    PP = 1
    max_turn = 5
    epsilon = 0.01
    #Retrieve the problem model after splitting and create a new folder named "ACP_Pickle"
    file_path = os.path.split(os.path.split(lp_file)[0])[0]
    pickle_path = os.path.join(file_path, 'ACP_Pickle')
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    max_turn_time = max_turn_ratio * time_limit
    n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem(lp_file)    
    # Build a variable dictionar
    num_to_value = {value : key for key, value in value_to_num.items()}

    #Get the start time
    begin_time = time.time()
    
    #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
    ans, ansx = get_ans(lp_file, time_limit, value_to_num)
    print(f"初始解目标值为：{ans}")
    
    #Constraint block labels, where cons_color[i] represents which block the i-th constraint belongs to
    cons_color = np.zeros(m, int)
    
    #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
    last = ans
    
    #Initialize the number of rounds below the threshold, which can be either 0 or 1 with little difference.
    turn = 1
    #Continue to the next iteration as long as it hasn't reached the maximum running time
    while(time.time() - begin_time <= time_limit):
        print("KK = ", KK)
        print("PP = ", PP)
        #Randomly divide the decision variables into KK blocks
        for i in range(m):
            cons_color[i] = random.randint(1, KK)
        now_cons_color = 1
        #Set all decision variables involved in a randomly selected constraint block as variables to be optimized
        #color[i] = 1 indicates that the i-th decision variable is selected as a variable to be optimized
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
        #site_to_color[i]represents which decision variable is the i-th decision variable in this block
        #color_to_site[i]represents which decision variable is mapped to the i-th decision variable in this block
        #vertex_color_num represents the number of decision variables in this block currently
        site_to_color = np.zeros(n, int)
        color_to_site = np.zeros(n, int)
        vertex_color_num = 0
        #Define the model to solve
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
        #Define decision variables x[]
        #x = model.addMVar((vertex_color_num), lb = 0, ub = 1, vtype = GRB.BINARY)  #lb is the lower bound for the variable， ub is the upper bound for the variables
        #Set up the objective function and optimization objective (maximization/minimization), only optimizing the variables selected for optimization
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
        #Add m constraints, only adding those constraints that involve variables in this block
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
        
        #Set the maximum solving time
        model.setParam('TimeLimit', min(max(time_limit - (time.time() - begin_time), 0), max_turn_time))
        #Optimize
        model.optimize()
        
        try:
            #Calculate the current objective value
            temp = model.ObjVal + objtemp
            print(f"当前目标值为：{temp}")
            bestX = []
            for i in range(vertex_color_num):
                bestX.append(x[i].X)
            #print(bestX)

            if(obj_type == 'maximize'):
                #Update the current best solution and best ans
                if(temp > ans):
                    for i in range(vertex_color_num):
                        ansx[color_to_site[i]] = bestX[i]
                    ans = temp
                #Adaptive block number change
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
                #Adaptive block number change
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
    new_ansx = {}
    for i in range(len(ansx)):
        new_ansx[num_to_value[i]] = ansx[i]
    with open(pickle_path + '/' + (os.path.split(lp_file)[1])[:-3] + '.pickle', 'wb') as f:
        pickle.dump([new_ansx, ans], f)
    return ans, time.time()-begin_time

def solve_LPs(file_paths, working_txt):
    '''
    Input the file paths stored in a list and solve the lp files in the 'LP' folder using ACP.
    '''
    for file_path in file_paths:
        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path}.\n')
        lp_folder = os.path.join(file_path, 'LP')
        lp_names = sorted(file for file in os.listdir(lp_folder) if file.endswith('.lp'))
        results = []
        # alive_bar
        with alive_bar(len(lp_names), title = file_path) as bar:
            for lp_name in lp_names:
                lp_file = os.path.join(lp_folder, lp_name)
                print(f'Solving {lp_file}!!')
                ans, s_time = run_ACP(lp_file, *get_ACP_params(os.path.split(file_path)[1]))
                results.append({'Filename': lp_name,
                                'Answer': ans,
                                'Solve_time': s_time})
                df = pd.DataFrame(results)
                # save{file_name，gap，solving_time} as A_GUROBI_output.xlsx in file_path
                df.to_excel(file_path+'/A_ACP_output.xlsx',index = False)
                print(f'Done {file_path}!')
                bar()

        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path} is end!\n')

def get_ACP_params(folder_name):
    '''
    根据数据集文件夹名返回参数
    '''
    time_limit = 4000
    block = 4
    max_turn_ratio = 0.01
    if folder_name == 'MIPlib':
        time_limit = 150
        block = 2
        max_turn_ratio = 0.1
    elif folder_name == 'CORAL':
        time_limit = 4000
        block = 2
    elif folder_name == 'Cut':
        time_limit = 4000
        block = 2
        max_turn_ratio = 0.05
    elif folder_name == 'ECOGCNN':
        time_limit = 4000
        block = 2
    elif folder_name == 'miplib_mixed_neos':
        time_limit = 4000
        block = 3
    elif folder_name == 'miplib_mixed_supportcase':
        time_limit = 4000
        block = 3
    elif folder_name == '1_item_placement':
        time_limit = 4000
        block = 3   
    elif folder_name == '2_load_balancing':
        time_limit = 1000
        block = 2
    elif folder_name == '3_anonymous':
        time_limit = 4000
        block = 2
    elif folder_name == 'Nexp':
        time_limit = 4000
        block = 3
    elif folder_name == 'Transportation':
        time_limit = 4000
        block = 3
    return block, time_limit, max_turn_ratio


working_txt = '/home/sharing/disk3/A_working.txt'
solve_LPs(find_all('/home/sharing/disk3/instance_folder'), working_txt)