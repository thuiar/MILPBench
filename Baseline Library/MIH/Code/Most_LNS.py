import os
import copy
import time
import gurobipy as gp
import numpy as np
from gurobipy import GRB, read, Model
import pandas as pd

def get_ans(lp_file, time_limit):
    model = gp.read(lp_file)
    model.Params.PoolSolutions = 1
    model.Params.PoolSearchMode = 1
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', 20)
    model.optimize()
    n = model.NumVars
    new_sol = np.zeros(n)

    try:
        for i in range(n):
            new_sol[i] = model.getVars()[i].X
    except:
        if(model.ModelSense == -1):
            for i in range(n):
                new_sol[i] = model.getVars()[i].UB
        else:
            for i in range(n):
                new_sol[i] = model.getVars()[i].LB
        
    return new_sol

def initial_solution(n, m, k, site, value, constraint, constraint_type, lower_bound, upper_bound, value_type):
    model = Model("Gurobi")
    x = []
    for i in range(n):
        if(value_type[i] == 'B'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.BINARY))
        elif(value_type[i] == 'C'):
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
        else:
            x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.INTEGER))
    model.setObjective(0, GRB.MAXIMIZE)
    for i in range(m):
        constr = 0
        for j in range(k[i]):
            constr += x[site[i][j]] * value[i][j]

        if(constraint_type[i] == 1):
            model.addConstr(constr <= constraint[i])
        elif(constraint_type[i] == 2):
            model.addConstr(constr >= constraint[i])
        else:
            model.addConstr(constr == constraint[i])
    model.optimize()
    new_sol = np.zeros(n)
    for i in range(n):
        if(value_type[i] == 'C'):
            new_sol[i] = x[i].X
        else:
            new_sol[i] = (int)(x[i].X)
        
    return new_sol

def initial_LP_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type):
    begin_time = time.time()
    model = Model("Gurobi")
    x = []
    for i in range(n):
        x.append(model.addVar(lb = lower_bound[i], ub = upper_bound[i], vtype = GRB.CONTINUOUS))
    coeff = 0
    for i in range(n):
        coeff += x[i] * coefficient[i]
    if(obj_type == -1):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
    for i in range(m):
        constr = 0
        for j in range(k[i]):
            constr += x[site[i][j]] * value[i][j]

        if(constraint_type[i] == 1):
            model.addConstr(constr <= constraint[i])
        elif(constraint_type[i] == 2):
            model.addConstr(constr >= constraint[i])
        else:
            model.addConstr(constr == constraint[i])
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    model.optimize()
    new_sol = np.zeros(n)
    try:
        for i in range(n):
            new_sol[i] = x[i].X
    except:
        if(obj_type == -1):
            for i in range(n):
                new_sol[i] = lower_bound[i]
        else:
            for i in range(n):
                new_sol[i] = upper_bound[i]
        
    return new_sol

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type, now_sol, now_col):
    """
    Function Description:
    Solve the given problem instance using the Gurobi solver.

    Parameter Description:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable is the j-th variable in the i-th constraint.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 indicates <= and 2 indicates >=.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    """

    begin_time = time.time()
    model = Model("Gurobi")
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
    
    coeff = 0
    for i in range(n):
        if(now_col[i] == 1):
            coeff += x[site_to_new[i]] * coefficient[i]
        else:
            coeff += now_sol[i] * coefficient[i]
    if(obj_type == -1):
        model.setObjective(coeff, GRB.MAXIMIZE)
    else:
        model.setObjective(coeff, GRB.MINIMIZE)
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
    #model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    model.optimize()
    try:
        new_sol = np.zeros(n)
        for i in range(n):
            if(now_col[i] == 0):
                new_sol[i] = now_sol[i]
            else:
                if(value_type[i] == 'C'):
                    new_sol[i] = x[site_to_new[i]].X
                else:
                    new_sol[i] = (int)(x[site_to_new[i]].X)
            
        return new_sol, model.ObjVal, 1
    except:
        return -1, -1, -1

def eval(n, coefficient, new_sol):
    ans = 0
    for i in range(n):
        ans += coefficient[i] * new_sol[i]
    return(ans)



def select_neighborhood(n, current_solution, lp_solution):
    neighbor_score = np.zeros(n)
    for var_index in range(n):
        neighbor_score[var_index] = -abs(current_solution[var_index] - lp_solution[var_index])
    return neighbor_score

def greedy_one(now_instance_data, time_limit):
    begin_time = time.time()
    set_time = time_limit
    epsilon = 1e-3
    n = now_instance_data[0]
    m = now_instance_data[1]
    k = now_instance_data[2]
    site = now_instance_data[3]
    value = now_instance_data[4] 
    constraint = now_instance_data[5]
    constraint_type = now_instance_data[6] 
    coefficient = now_instance_data[7]
    obj_type = now_instance_data[8]
    lower_bound = now_instance_data[9]
    upper_bound = now_instance_data[10]
    value_type = now_instance_data[11]
    initial_sol = now_instance_data[12]

    choose = 0.5
    best_val = eval(n, coefficient, initial_sol)
    
    turn_time = [time.time() - begin_time]
    turn_ans = [best_val]

    #Find LP solution
    LP_sol = initial_LP_solution(n, m, k, site, value, constraint, constraint_type, coefficient, set_time * 0.3, obj_type, lower_bound, upper_bound, value_type)
    
    turn_limit = 100
    
    now_sol = initial_sol
    while(time.time() - begin_time <= set_time):
        #print("before", parts, time.time() - begin_time)
        #"n", "m", "k", "site", "value", "constraint", "initial_solution", "current_solution", "objective_coefficient"
        neighbor_score = select_neighborhood(
                            n, 
                            copy.deepcopy(now_sol), 
                            copy.deepcopy(LP_sol)
                        )
        #print("after", parts, time.time() - begin_time)
        indices = np.argsort(neighbor_score)[::-1]
        color = np.zeros(n)
        for i in range(int(n * choose)):
            color[indices[i]] = 1
        new_sol, now_val, now_flag = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(set_time - (time.time() - begin_time), turn_limit), obj_type, lower_bound, upper_bound, value_type, now_sol, color)
        if(now_flag == -1):
            continue
        
        #Maximize
        if(obj_type == -1):
            if(now_val > best_val):
                now_sol = new_sol
                best_val = now_val
        else:
            if(now_val < best_val):
                now_sol = new_sol
                best_val = now_val

        turn_ans.append(best_val) 
        turn_time.append(time.time() - begin_time)
    return(turn_ans, turn_time)

def split_problem(lp_file):
    """
    Function Description:
    Solve the given problem instance using the Gurobi solver.

    Parameter Description:
    - n: The number of decision variables in the problem instance.
    - m: The number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable is the j-th variable in the i-th constraint.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 indicates <= and 2 indicates >=.
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: The maximum solving time.
    - obj_type: Indicates whether the problem is a maximization or minimization problem.
    """
    model = gp.read(lp_file)
    n = model.NumVars
    m = model.NumConstrs
    k = []
    site = []
    value = []
    constraint = []
    constraint_type = []
    coefficient = []
    obj_type = model.ModelSense
    upper_bound = []
    lower_bound = []
    value_type = []
    var_name_to_index = {}

    objective = model.getObjective()
    temp_coeff = []
    temp_varname = []
    for i in range(objective.size()):
        temp_coeff.append(objective.getCoeff(i))
        temp_varname.append(objective.getVar(i).VarName)

    i = 0
    for var in model.getVars():
        var_name_to_index[var.VarName] = i
        upper_bound.append(var.UB)
        lower_bound.append(var.LB)
        value_type.append(var.VType)
        if var.VarName not in temp_varname:
            coefficient.append(0)
        else:
            coefficient.append(temp_coeff[temp_varname.index(var.VarName)])
        i+=1

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
            now_site.append(var_name_to_index[row.getVar(i).VarName])
            now_value.append(row.getCoeff(i))
        site.append(now_site)
        value.append(now_value)
        
    return n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type

def run_Most_LNS(lp_file, time_limit):
    n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, lower_bound, upper_bound, value_type = split_problem(lp_file)
    begin_time = time.time()
    new_sol = get_ans(lp_file, time_limit)
    time_limit = time_limit - (time.time() - begin_time)
    new_site = []
    new_value = []
    new_constraint = np.zeros(m)
    new_constraint_type = np.zeros(m, int)
    for i in range(m):
        new_site.append(np.zeros(k[i], int))
        new_value.append(np.zeros(k[i]))
        for j in range(k[i]):
            new_site[i][j] = site[i][j]
            new_value[i][j] = value[i][j]
        new_constraint[i] = constraint[i]
        new_constraint_type[i] = constraint_type[i]
    
    new_coefficient = np.zeros(n)
    new_lower_bound = np.zeros(n)
    new_upper_bound = np.zeros(n)
    new_value_type = np.zeros(n, int)
    new_new_sol = np.zeros(n)
    for i in range(n):
        new_coefficient[i] = coefficient[i]
        new_lower_bound[i] = lower_bound[i]
        new_upper_bound[i] = upper_bound[i]
        if(value_type[i] == 'B'):
            new_value_type[i] = 0
        elif(value_type[i] == 'C'):
            new_value_type[i] = 1
        else:
            new_value_type[i] = 2
        new_new_sol[i] = new_sol[i]

    now_instance = (n, m, k, new_site, new_value, new_constraint, new_constraint_type, new_coefficient, obj_type, new_lower_bound, new_upper_bound, new_value_type, new_new_sol)
    now_sol, now_time = greedy_one(now_instance, time_limit)
    # print(now_sol)
    # print(now_time)
    return now_sol, now_time


def solve_LPs(file_paths, working_txt):
    '''
    Pass a list of file paths stored in a list, use SCIP to solve the LP files in the `lp` folder within them, and save the working log in `working_txt`.
    '''
    for file_path in file_paths:
        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path}.\n')
        lp_folder = os.path.join(file_path, 'LP')
        lp_names = sorted(file for file in os.listdir(lp_folder) if file.endswith('.lp'))
        results = []
        for lp_name in lp_names:
            lp_file = os.path.join(lp_folder, lp_name)
            print(f'Solving {lp_file}!!')
            ans, s_time = run_Most_LNS(lp_file, get_params(os.path.split(file_path)[1]))
            results.append({'Filename': lp_name,
                            'Answer': ans,
                            'Solve_time': s_time})
            df = pd.DataFrame(results)
            # Save `{filename, gap, solving time}` in `A_Least)LNS_output.xlsx` under the file path.
            df.to_excel(file_path+'/A_Most_LNS_output.xlsx',index = False)
            print(f'Done {file_path}!')

        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path} is end!\n')

def find_all(root_folder, string= 'LP'):
    '''
    Return the folder paths under `root_folder` that contain files with 'LP' in their name and store them in a list.
    '''
    file_paths = []
    for root, dir, files in os.walk(root_folder):
        if string in dir:
            file_paths.append(root)
    return(file_paths)

def sort_by_custom_order(list_a, list_b):
    """
    Sort list_a based on the order of elements in list_b.

    Args:
        list_a (list): The list to be sorted.
        list_b (list): The list defining the sort order.

    Returns:
        list: The sorted list_a.
    """
    order_map = {key: index for index, key in enumerate(list_b)}
    sorted_list = sorted(list_a, key=lambda item: next((order_map[word] for word in list_b if word in item), float('inf')))
    
    return sorted_list

def list_filter(long_list, short_list):
    """
    Remove elements from the first list that contain any of the strings in the second list.
    :param long_list: list[str], the longer list of strings
    :param short_list: list[str], the shorter list of strings
    :return: list[str], the filtered first list
    """
    # Iterate through the longer string list and filter out items that contain any of the shorter strings.
    filtered_list = [long_str for long_str in long_list 
                     if not any(short_str in long_str for short_str in short_list)]
    return filtered_list

def get_params(folder_name):
    '''
    Return parameters based on the dataset folder name.
    '''
    # if 'easy' in folder_name:
    #     time_limit = 600
    # elif 'medium' in folder_name:
    #     time_limit = 2000
    # elif 'hard' in folder_name:
    #     time_limit = 4000

    if folder_name == 'Aclib':
        time_limit = 100
    elif folder_name == 'CORAL':
        time_limit = 4000
    elif folder_name == 'Cut':
        time_limit = 4000
    elif folder_name == 'ECOGCNN':
        time_limit = 4000
    elif folder_name == 'fc.data':
        time_limit = 100
    elif folder_name == 'knapsack':
        time_limit = 100
    elif folder_name == 'mis':
        time_limit = 100
    elif folder_name == 'setcover':
        time_limit = 100
    elif folder_name == 'corlat':
        time_limit = 100
    elif folder_name == 'mik':
        time_limit = 100
    elif folder_name == '1_item_placement':
        time_limit = 4000
    elif folder_name == '2_load_balancing':
        time_limit = 1000
    elif folder_name == '3_anonymous':
        time_limit = 4000
    elif folder_name == 'MIPlib':
        time_limit = 150
    elif folder_name == 'Nexp':
        time_limit = 4000
    elif folder_name == 'nn_verification':
        time_limit = 100
    elif folder_name == 'Transportation':
        time_limit = 4000
    elif folder_name == 'series_1':
        time_limit = 600
    elif folder_name == 'series_2':
        time_limit = 100
    elif folder_name == 'series_3':
        time_limit = 100
    elif folder_name == 'series_4':
        time_limit = 100
    elif '_easy_instance' in folder_name:
        time_limit = 600
    elif '_medium_instance'in folder_name:
        time_limit = 2000
    elif '_hard_instance' in folder_name:
        time_limit = 4000
    return time_limit

if __name__ == '__main__':
    
    # order_0 = ['Aclib', 'CORAL', 'Cut', 'ECOGCNN', 'fc.data', 'knapsack', 'mis', 'setcover', 'corlat', 'mik', '1_item_placement', '2_load_balancing', '3_anonymous', 'MIPlib', 'Nexp', 'nn_verification', 'Transportation']
    # filter_0 = ['Aclib', 'CORAL', 'Cut', 'ECOGCNN', 'fc.data', 'knapsack', 'mis', 'setcover', 'corlat', 'mik']
    working_txt = 'A_working.txt'
    # dataset_folders = find_all('MIPcc23-Dataset/vary_rhs')
    dataset_folders = find_all('test_datasets')
    # sorted_folders = sort_by_custom_order(dataset_folders, order_0)
    # filted_folders = list_filter(sorted_folders, filter_0)

    solve_LPs(dataset_folders, working_txt)