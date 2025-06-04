import os
import gurobipy as gp
import pandas as pd
import time
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

def run_GUROBI(LP_file, time_limit):
    '''
    Pass in the paths of LP files, invoke the Gurobi solver for solving, save the solution and gap into a pickle file, and return the resulting gap value along with the time consumed for solving
    '''
    model = gp.read(LP_file)
    model.setParam('TimeLimit', time_limit)
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    s_time = end_time - start_time
    # save the solution into pickle file in GUROBI_pickle folder
    file_path = os.path.split(os.path.split(LP_file)[0])[0]
    pickle_path = os.path.join(file_path, 'GUROBI_Pickle')
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    ans = {}
    try:
        for val in model.getVars():
            ans[val.VarName] = val.x
        with open(pickle_path + '/' + (os.path.split(LP_file)[1])[:-3] + '.pickle', 'wb') as f:
            pickle.dump([ans, model.MIPGap], f)
    except:
        if not model.IsMIP:
            for val in model.getVars():
                ans[val.VarName] = val.x
            with open(pickle_path + '/' + (os.path.split(LP_file)[1])[:-3] + '.pickle', 'wb') as f:
                pickle.dump([ans, 0], f)
            return(0, s_time)
        else:
            with open(pickle_path + '/' + (os.path.split(LP_file)[1])[:-3] + '.pickle', 'wb') as f:
                pickle.dump([None, None], f)
            return(None, s_time)
    return(model.MIPGap, s_time)

def solve_LPs(file_paths, working_txt):
    '''
    Input the file paths stored in a list and solve the lp files in the 'LP' folder using Gurobi.
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
                gap, s_time = run_GUROBI(lp_file, get_time_limit(os.path.split(file_path)[1]))
                results.append({'Filename': lp_name,
                                'MIPGap': gap,
                                'Solve_time': s_time})
                df = pd.DataFrame(results)
                # save{file_name，gap，solving_time} as A_GUROBI_output.xlsx in file_path
                df.to_excel(file_path+'/A_GUROBI_output.xlsx',index = False)
                print(f'Done {file_path}!')
                bar()

        with open(working_txt, 'a', encoding='utf-8')as f:
            f.write(f'Working in {file_path} is end!\n')

def get_time_limit(folder_name):
    '''
    Return time_limit based on the problem folder name
    '''
    time_limit = 100
    if folder_name == 'MIPlib':
        time_limit = 150
    elif folder_name == 'CORAL':
        time_limit = 4000
    elif folder_name == 'Cut':
        time_limit = 4000
    elif folder_name == 'ECOGCNN':
        time_limit = 4000
    elif folder_name == 'miplib_mixed_neos':
        time_limit = 4000
    elif folder_name == 'miplib_mixed_supportcase':
        time_limit = 4000
    elif folder_name == '1_item_placement':
        time_limit = 4000
    elif folder_name == '2_load_balancing':
        time_limit = 1000
    elif folder_name == '3_anonymous':
        time_limit = 4000
    elif folder_name == 'Nexp':
        time_limit = 4000
    elif folder_name == 'Transportation':
        time_limit = 4000
    return time_limit

working_txt = '/home/sharing/disk3/A_working.txt'
solve_LPs(find_all('/home/sharing/disk3/instance_folder'), working_txt)