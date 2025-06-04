import os
import shutil
import random

def dataset(instance_folder):
    '''
    Pass the instance folder directory, which should contain three folders: test, train, and valid
    '''
    test_folder = instance_folder+'/test'
    train_folder = instance_folder+'/train'
    valid_folder = instance_folder+'/valid'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)
    test_names = sorted(f for f in os.listdir(test_folder)if f.endswith('.lp'))
    for test_name in test_names:
        try:
            os.remove(f'{train_folder}/{test_name}')
        except:
            print(f'No {train_folder}/{test_name}...')
    train_names = sorted(f for f in os.listdir(train_folder)if f.endswith('.lp'))
    train_files = []
    for train_name in train_names:
        train_files.append(os.path.join(train_folder, train_name))
    valid_files = random.sample(train_files, int(0.2*len(train_files)))
    for valid_file in valid_files:
        shutil.move(valid_file, valid_folder)
        print(f'Moved {valid_file} to {valid_folder}')



for problem in   ['1_item_placement',
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
                 'vary_rhs_s4',]:
    dataset(f'data/instances/{problem}')
