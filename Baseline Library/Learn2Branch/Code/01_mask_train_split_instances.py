import os
import shutil
import random

def dataset(instance_folder):
    '''
    Enter the instance folder directory, which should contain three folders: test, train, and valid
    '''
    test_folder = instance_folder+'/test'
    train_folder = instance_folder+'/train'
    valid_folder = instance_folder+'/valid'
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



dataset('data/instances/MVC_easy_instance')
