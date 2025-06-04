import os
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import time
import random
import pickle
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

def get_ans(lp_file, max_turn_time, value_to_num):
    model = gp.read(lp_file)
    model.Params.PoolSolutions = 1
    model.Params.PoolSearchMode = 1
    model.setParam('TimeLimit', max_turn_time)
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

def run_LNS(lp_file, block = 2, time_limit = 4000, max_turn_ratio = 0.1):
    '''
    Run LNS (Large Neighborhood Search), passing in the lp file. 'block' is the number of blocks (default 2), 'time_limit' is the total running time limit (default 4000), and 'max_turn_ratio' is the maximum running time ratio for each turn (default 0.1).
    '''
    file_path = os.path.split(os.path.split(lp_file)[0])[0]
    pickle_path = os.path.join(file_path, 'LNS_Pickle')
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    max_turn_time = max_turn_ratio * time_limit
    n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type, value_to_num = split_problem(lp_file)
    # Build a variable dictionar
    num_to_value = {value : key for key, value in value_to_num.items()}
    #Set KK as the initial number of blocks, and PP as the selected number of blocks to optimize after dividing the constraints into KK blocks
    KK = block
    #Get the start time
    begin_time = time.time()
    #Initialize the initial solution and initial answer, where the initial solution is the worst initial solution
    ans, ansx = get_ans(lp_file, time_limit, value_to_num)
    print(f"Initial objective: {ans}")
    #Continue to the next iteration as long as it hasn't reached the maximum running time
    while(time.time() - begin_time <= time_limit):
        print("KK = ", KK)
        #Randomly divide the decision variables into KK blocks
        color = np.zeros(n, int)
        for i in range(n):
            color[i] = random.randint(1, KK)
        #Enumerate each of the blocks
        for now_color in range(1, KK + 1):
            #Exit when reaching the time limit
            if(time.time() - begin_time > time_limit):
                break
            #site_to_color[i]represents which decision variable is the i-th decision variable in this block
            #color_to_site[i]represents which decision variable is mapped to the i-th decision variable in this block
            #vertex_color_num represents the number of decision variables in this block currently
            site_to_color = np.zeros(n, int)
            color_to_site = np.zeros(n, int)
            vertex_color_num = 0

            #Define the model to solve
            model = gp.Model("LNS")
            #Define decision variables x[]
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
                
            #Set up the objective function and optimization objective (maximization/minimization), only optimizing the variables in this block
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
                print(f"The current objective value is: {temp}")
                bestX = []
                for i in range(vertex_color_num):
                    bestX.append(x[i].X)
                #print(bestX)

                #Update the current best solution and best ans
                if(obj_type == 'maximize'):
                    if(temp > ans):
                        for i in range(vertex_color_num):
                            ansx[color_to_site[i]] = bestX[i]
                        ans = temp
                else:
                    if(temp < ans):
                        for i in range(vertex_color_num):
                            ansx[color_to_site[i]] = bestX[i]
                        ans = temp
            except:
                print("Cant't optimize more~~")
                # new_ansx = {}
                # for i in range(len(ansx)):
                #     new_ansx[num_to_value[i]] = ansx[i]
                # with open(pickle_path + '/' + (os.path.split(lp_file)[1])[:-3] + '.pickle', 'wb') as f:
                #     pickle.dump([new_ansx, ans], f)
                # return ans, time.time()-begin_time
        
    new_ansx = {}
    for i in range(len(ansx)):
        new_ansx[num_to_value[i]] = ansx[i]
    with open(pickle_path + '/' + (os.path.split(lp_file)[1])[:-3] + '.pickle', 'wb') as f:
        pickle.dump([new_ansx, ans], f)
    return ans, time.time()-begin_time

def solve_LPs(file_paths, working_txt):
    '''
    Input the file paths stored in a list and solve the lp files in the 'LP' folder using LNS.
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
                ans, s_time = run_LNS(lp_file, *get_LNS_params(os.path.split(file_path)[1]))
                results.append({'Filename': lp_name,
                                'Answer': ans,
                                'Solve_time': s_time})
                df = pd.DataFrame(results)
                # save{file_name，gap，solving_time} as A_GUROBI_output.xlsx in file_path
                df.to_excel(file_path+'/A_LNS_output.xlsx',index = False)
                print(f'Done {file_path}!')
                bar()

        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path} is end!\n')

def get_LNS_params(folder_name):
    '''
    Return time_limit based on the problem folder name
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
# run_LNS(r'MVC\LP\MVC_medium_instance_0.lp', 2, 60)
