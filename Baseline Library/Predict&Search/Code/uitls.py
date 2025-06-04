import os
import shutil

def delete_test_from_train(TaskName:str):
    test_folder = f'instance/test/{TaskName}'
    test_names = sorted(f for f in os.listdir(test_folder)if f.endswith('.lp'))

    train_folder = f'instance/train/{TaskName}'
    train_names = sorted(f for f in os.listdir(train_folder)if f.endswith('.lp'))

    for train_name in train_names:
        if train_name in test_names:
            os.remove(os.path.join(train_folder, train_name))
            print(f'removed {os.path.join(train_folder, train_name)}')

tasknames = ['vary_matrix_rhs_bounds_obj_s1', 'vary_matrix_rhs_bounds_s1', 'vary_matrix_s1', 'vary_obj_s1', 'vary_obj_s2', 'vary_obj_s3', 'vary_rhs_obj_s1','vary_rhs_obj_s2','vary_rhs_s1','vary_rhs_s2','vary_rhs_s3','vary_rhs_s4']

for taskname in tasknames:
    delete_test_from_train(taskname)