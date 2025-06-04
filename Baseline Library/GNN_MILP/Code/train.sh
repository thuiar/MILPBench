



# nohup ./train.sh >output.log 2>&1 &

# python my1_generate_data_new.py knapsack
python my2_training_new.py knapsack
python my3_testing_new.py knapsack

# python my1_generate_data_new.py mis
python my2_training_new.py mis
python my3_testing_new.py mis

# python my1_generate_data_new.py setcover
python my2_training_new.py setcover
python my3_testing_new.py setcover

# python my1_generate_data_new.py vary_bounds_s1
python my2_training_new.py vary_bounds_s1
python my3_testing_new.py vary_bounds_s1

# python my1_generate_data_new.py vary_matrix_rhs_bounds_obj_s1
python my2_training_new.py vary_matrix_rhs_bounds_obj_s1
python my3_testing_new.py vary_matrix_rhs_bounds_obj_s1

# python my1_generate_data_new.py vary_matrix_s1
python my2_training_new.py vary_matrix_s1
python my3_testing_new.py vary_matrix_s1

# python my1_generate_data_new.py vary_obj_s1
python my2_training_new.py vary_obj_s1
python my3_testing_new.py vary_obj_s1

# python my1_generate_data_new.py vary_obj_s3
python my2_training_new.py vary_obj_s3
python my3_testing_new.py vary_obj_s3

# python my1_generate_data_new.py vary_rhs_obj_s2
python my2_training_new.py vary_rhs_obj_s2
python my3_testing_new.py vary_rhs_obj_s2

# python my1_generate_data_new.py vary_rhs_s2
python my2_training_new.py vary_rhs_s2
python my3_testing_new.py vary_rhs_s2

# python my1_generate_data_new.py vary_rhs_s4
python my2_training_new.py vary_rhs_s4
python my3_testing_new.py vary_rhs_s4

# python my1_generate_data_new.py MIPlib
python my2_training_new.py MIPlib
python my3_testing_new.py MIPlib