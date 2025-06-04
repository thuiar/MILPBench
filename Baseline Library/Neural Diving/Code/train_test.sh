# nohup ./train_test.sh >ND2output.log 2>&1 &
# 549639

# python 1_read_lp.py train knapsack
# python 2_train.py knapsack
# python 1_read_lp.py test knapsack
# python 3_test.py knapsack

# python 1_read_lp.py train mis
python 2_train.py mis
# python 1_read_lp.py test mis
python 3_test.py mis

# python 1_read_lp.py train setcover
python 2_train.py setcover
# python 1_read_lp.py test setcover
python 3_test.py setcover

# python 1_read_lp.py train vary_bounds_s1
python 2_train.py vary_bounds_s1
# python 1_read_lp.py test vary_bounds_s1
python 3_test.py vary_bounds_s1

# python 1_read_lp.py train vary_matrix_rhs_bounds_obj_s1
python 2_train.py vary_matrix_rhs_bounds_obj_s1
# python 1_read_lp.py test vary_matrix_rhs_bounds_obj_s1
python 3_test.py vary_matrix_rhs_bounds_obj_s1

# python 1_read_lp.py train vary_matrix_s1
python 2_train.py vary_matrix_s1
# python 1_read_lp.py test vary_matrix_s1
python 3_test.py vary_matrix_s1

# python 1_read_lp.py train vary_obj_s1
python 2_train.py vary_obj_s1
# python 1_read_lp.py test vary_obj_s1
python 3_test.py vary_obj_s1

# python 1_read_lp.py train vary_obj_s3
python 2_train.py vary_obj_s3
# python 1_read_lp.py test vary_obj_s3
python 3_test.py vary_obj_s3

# python 1_read_lp.py train vary_rhs_obj_s2
python 2_train.py vary_rhs_obj_s2
# python 1_read_lp.py test vary_rhs_obj_s2
python 3_test.py vary_rhs_obj_s2

# python 1_read_lp.py train vary_rhs_s2
python 2_train.py vary_rhs_s2
# python 1_read_lp.py test vary_rhs_s2
python 3_test.py vary_rhs_s2

# python 1_read_lp.py train vary_rhs_s4
python 2_train.py vary_rhs_s4
# python 1_read_lp.py test vary_rhs_s4
python 3_test.py vary_rhs_s4
