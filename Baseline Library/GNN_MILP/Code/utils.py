import shutil
import os

folder = 'instances'
# names = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder,f))]

# for name in names:
#     problem_folder = os.path.join(folder, name)
#     train_folder = f'{problem_folder}/train'
#     valid_folder = f'{problem_folder}/valid'
#     test_folder = f'{problem_folder}/test'
#     for f in os.listdir(valid_folder):
#         shutil.move(os.path.join(valid_folder, f), os.path.join(train_folder, f))

#     os.removedirs(valid_folder)
#     os.rename(train_folder, f'{problem_folder}/LP')
#     os.rename(test_folder, f'{problem_folder}/LP_test')


# for root, dirs, files in os.walk(folder):
#     for f in files:
#         if f.endswith('.md'):
#             os.remove(os.path.join(root, f))
#             print(f'removed {os.path.join(root, f)}')