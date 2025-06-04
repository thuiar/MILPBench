import os
import cplex
import numpy as np
import pandas as pd
import time
import pickle
from cplex.exceptions import CplexError

def find_all(root_folder, string= 'Pickle'):
    '''
    Return the folder paths under `root_folder` that contain files with 'LP' in their name and store them in a list.
    '''
    file_paths = []
    for root, dir, files in os.walk(root_folder):
        if string in dir:
            file_paths.append(root)
    return(file_paths)

def run_CPLEX(LP_file, time_limit = 4000):
    '''
    Pass the LP file path to the GUROBI solver for solving, store the results and the gap in a pickle file, and return the gap value and the solving time.
    The default time limit is 4000 seconds.
    '''
    model = cplex.Cplex()
    model.read(LP_file)
    model.parameters.timelimit.set(time_limit)
    start_time = time.time()
    model.solve()
    end_time = time.time()
    s_time = end_time - start_time
    # Store the results in pickle format under the `GUROBI_pickle` folder.
    file_path = os.path.split(os.path.split(LP_file)[0])[0]
    pickle_path = os.path.join(file_path, 'CPLEX_Pickle')
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    ans = {}
    try:
        obj_value = model.solution.get_objective_value()
        mip_gap = model.solution.MIP.get_mip_relative_gap()
        variable_values = model.solution.get_values()
        for var_name, var_value in zip(model.variables.get_names(), variable_values):
            ans[var_name] = var_value
        with open(pickle_path + '/' + (os.path.split(LP_file)[1])[:-3] + '.pickle', 'wb') as f:
            pickle.dump([ans, mip_gap], f)
    except:
        return(None, None, s_time)
    return(obj_value, mip_gap, s_time)

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
            obj_value, gap, s_time = run_CPLEX(lp_file, get_time_limit(os.path.split(file_path)[1]))
            results.append({'Filename': lp_name,
                            'Objective': obj_value,
                            'MIPGap': gap,
                            'Solve_time': s_time})
            df = pd.DataFrame(results)
            # Save `{filename, gap, solving time}` in `A_CPLEX_output.xlsx` under the file path.
            df.to_excel(file_path+'/A_CPLEX_output.xlsx',index = False)
            print(f'Done {file_path}!')

        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path} is end!\n')

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

def get_time_limit(folder_name):
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
    elif '_easy_instance' in folder_name:
        time_limit = 600
    elif '_medium_instance'in folder_name:
        time_limit = 2000
    elif '_hard_instance' in folder_name:
        time_limit = 4000
    # # vary_bounds
    # elif folder_name == 'series_1':
    #     time_limit = 400
    # elif folder_name == 'series_2':
    #     time_limit = 1000
    # elif folder_name == 'series_3':
    #     time_limit = 1000
    # # vary_matrix; vary_matrix_rhs_bounds_s1, vary_matrix_rhs_bounds_obj_s1
    # elif folder_name == 'series_1':
    #     time_limit = 100
    # vary_obj
    # elif folder_name == 'series_1':
    #     time_limit = 100
    # elif folder_name == 'series_2':
    #     time_limit = 150
    # elif folder_name == 'series_3':
    #     time_limit = 100
    # # vary_rhs
    # elif folder_name == 'series_1':
    #     time_limit = 100
    # elif folder_name == 'series_2':
    #     time_limit = 100
    # elif folder_name == 'series_3':
    #     time_limit = 100
    # elif folder_name == 'series_4':
    #     time_limit = 100
    # vary_rhs_obj
    elif folder_name == 'series_1':
        time_limit = 600
    elif folder_name == 'series_2':
        time_limit = 100
    return time_limit



working_txt = 'A_working.txt'
# order_0 = ['Aclib', 'CORAL', 'Cut', 'ECOGCNN', 'fc.data', 'knapsack', 'mis', 'setcover', 'corlat', 'mik', '1_item_placement', '2_load_balancing', '3_anonymous', 'MIPlib', 'Nexp', 'nn_verification', 'Transportation']
# filter_0 = ['Aclib', 'CORAL', 'Cut', 'ECOGCNN', 'fc.data', 'knapsack', 'mis', 'setcover', 'corlat', 'mik']
dataset_folders = find_all('test_datasets')
# sorted_folders = sort_by_custom_order(dataset_folders, order_0)
# filted_folders = list_filter(sorted_folders, filter_0)

solve_LPs(dataset_folders, working_txt)