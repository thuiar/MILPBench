# tensorflow=2.4
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
import time
from gurobipy import *
from models import GCNPolicy

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col, lower_bound, upper_bound, value_type):
#def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, now_col, constr_flag, lower_bound, upper_bound, value_type):
    '''
	Function Description:
	Use Gurobi solver to solve the problem based on the provided problem instance.

	Parameter description:
	-N: The number of decision variables in the problem instance.
	-M: The number of constraints for problem instances.
	-K: k [i] represents the number of decision variables for the i-th constraint.
	-Site: site [i] [j] represents which decision variable is the jth decision variable of the i-th constraint.
	-Value: value [i] [j] represents the coefficient of the jth decision variable of the i-th constraint.
	-Constraint: constraint [i] represents the number to the right of the i-th constraint.
	-Constrict_type: constrict_type [i] represents the type of the i-th constraint, 1 represents<=, 2 represents>=
	-Coefficient: coefficient [i] represents the coefficient of the i-th decision variable in the objective function.
	-Time_imit: Maximum solution time.
	-Obj_type: Is the problem a maximization problem or a minimization problem.
    '''
    begin_time = time.time()
    model = Model("Gurobi")
    model.feasRelaxS(0,False,False,True)
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
    if(obj_type == 'maximize'):
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
                    #print(now_col)
            else:
                if(constr < constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    #print(now_col)
    model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    model.optimize()
    #print(time.time() - begin_time)
    try:
        new_sol = []
        for i in range(n):
            if(now_col[i] == 0):
                new_sol.append(now_sol[i])
            else:
                if(value_type[i] == 'C'):
                    new_sol.append(x[site_to_new[i]].X)
                else:
                    new_sol.append((int)(x[site_to_new[i]].X))
            
        return 1, new_sol, model.ObjVal
    except:
        #model.computeIIS()
        #model.write("abc.ilp")
        return -1, -1, -1

def log(any, txt = 'GNN-MILP_result.txt'):
	with open(txt, 'a')as f:
		f.writelines(str(any))
		f.writelines('\n')

## ARGUMENTS OF THE SCRIPT
# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu", 		help="gpu index", 					default="0")
# parser.add_argument("--embSize", 	help="embedding size of GNN", 		default="6")
# parser.add_argument("--epoch", 		help="num of epoch", 				default="10000")
# parser.add_argument("--type", 		help="what's the type of the model",default="fea", 	choices = ['fea','obj','sol'])
# parser.add_argument("--data_path", 	help="path of data", 				default=None)
# args = parser.parse_args()

## FUNCTION OF TRAINING PER EPOCH

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'problem',
		choices=['setcover', 'cauctions', 'facilities', 'indset',
					'1_item_placement',
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
					'vary_rhs_s4',],
	)
	args = parser.parse_args()
	problem_name = args.problem

	# time_limit
	time_limit = 100
	if args.problem == '1_item_placement':
		time_limit = 4000
	elif args.problem == 'Cut':
		time_limit = 4000
	elif args.problem == 'MIPlib':
		time_limit = 150
	elif args.problem == 'miplib_mixed_neos':
		time_limit = 4000
	elif args.problem == 'Nexp':
		time_limit = 4000
	elif args.problem == 'Transportation':
		time_limit = 4000
	elif args.problem == 'vary_bounds_s1':
		time_limit = 400
	elif args.problem == 'vary_bounds_s2':
		time_limit = 1000
	elif args.problem == 'vary_bounds_s3':
		time_limit = 1000
	elif args.problem == 'vary_matrix_rhs_bounds_obj_s1':
		time_limit = 100
	elif args.problem == 'vary_matrix_s1':
		time_limit = 100
	elif args.problem == 'vary_obj_s1':
		time_limit = 100
	elif args.problem == 'vary_obj_s2':
		time_limit = 150
	elif args.problem == 'vary_obj_s3':
		time_limit = 100
	elif args.problem == 'vary_rhs_obj_s2':
		time_limit = 100
	elif args.problem == 'vary_rhs_s2':
		time_limit = 100
	elif args.problem == 'vary_rhs_s2':
		time_limit = 100
	elif args.problem == 'vary_rhs_s4':
		time_limit = 100
	## SET-UP DATASET
	trainfolder = f'instances/{problem_name}/train'
	testfolder = f'instances/{problem_name}/test'
	gpu_index = 1
	embSize = 6
	instance_names = lp_names = sorted(f for f in os.listdir(f'instances/{problem_name}/LP_test')if f.endswith('.lp'))

	## SET-UP MODEL
	if not os.path.exists('./saved-models/'):
		os.mkdir('./saved-models/')
	model_setting = trainfolder.replace('/','-')
	model_path = f'saved-models/.-instances-{problem_name}-train--obj-s6.pkl'

	## LOAD DATASET INTO MEMORY

	vars_all = 0
	cons_all = 0


	varFeatures = read_csv(testfolder + "/VarFeatures_feas.csv", header=None)
	conFeatures = read_csv(testfolder + "/ConFeatures_feas.csv", header=None)
	edgFeatures = read_csv(testfolder + "/EdgeFeatures_feas.csv", header=None)
	edgIndices = read_csv(testfolder + "/EdgeIndices_feas.csv", header=None)
	labels = read_csv(testfolder + "/Labels_solu.csv", header=None)
	n_Cons = read_csv(testfolder + "/Con_num.csv", header=None).values.reshape(-1)
	n_Vars = read_csv(testfolder + "/Var_num.csv", header=None).values.reshape(-1)
	nonzero = read_csv(testfolder + "/Nonzero_num.csv", header=None).values.reshape(-1)
	obj_type = read_csv(testfolder + "/Obj_type.csv", header=None).values.reshape(-1)

	nConsF = conFeatures.shape[1]
	nVarF = varFeatures.shape[1]
	nEdgeF = edgFeatures.shape[1]


	## SET-UP TENSORFLOW
	seed = 0
	tf.random.set_seed(seed)
	gpu_index = int(gpu_index)
	tf.config.set_soft_device_placement(True)
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
	tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

	model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
	model.restore_state(model_path)

	result = []
	repair_time = []
	with tf.device("GPU:"+str(gpu_index)):
		#print(varFeatures)
		### LOAD DATASET INTO GPU ###
		return_loss = 0
		now_n = 0
		now_m = 0
		now_nonzero = 0
		varFeatures_list = []
		conFeature_list = []
		edgFeatures_list = []
		edgIndices_list = []
		labels_list = []
		for now_num in range(len(n_Cons)):
			turn_time = time.time()
			now_varFeatures = tf.constant(varFeatures.values[now_n : now_n + n_Vars[now_num], :], dtype=tf.float32)
			now_conFeature = tf.constant(conFeatures.values[now_m : now_m + n_Cons[now_num], :], dtype=tf.float32)
			now_edgFeatures = tf.constant(edgFeatures.values[now_nonzero : now_nonzero + nonzero[now_num], :], dtype=tf.float32)
			now_edgIndices = tf.transpose(tf.constant(edgIndices.values[now_nonzero : now_nonzero + nonzero[now_num], :], dtype=tf.int32))

			now_n += n_Vars[now_num]
			now_m += n_Cons[now_num]
			now_nonzero += nonzero[now_num]

			test_data = (now_conFeature, now_edgIndices, now_edgFeatures, now_varFeatures, n_Cons[now_num], n_Vars[now_num], n_Cons[now_num], n_Vars[now_num])  	
			logits = model(test_data, tf.convert_to_tensor(True)).numpy().reshape(-1)
			#print(logits)

			#check the feasiblility of soltion
			n = n_Vars[now_num]
			m = n_Cons[now_num]
			k = []
			site = []
			value = []
			value_type = []
			constraint_value = []
			constraint_type = []
			coefficient = []
			lower_bound = []
			upper_bound = []

			feasible_flag = 1
			eps = 1e-6

			for i in range(n):
				coefficient.append(now_varFeatures[i][0].numpy().item())
				value_type.append(now_varFeatures[i][1].numpy().item())
				lower_bound.append(now_varFeatures[i][2].numpy().item())
				upper_bound.append(now_varFeatures[i][3].numpy().item())

				if(value_type[i] != 0):
					logits[i] = int(logits[i] + 0.5)
					logits[i] = min(logits[i], upper_bound[i])
					logits[i] = max(logits[i], lower_bound[i])
				
				if(value_type[i] == 0):
					value_type[i] = 'C'
				if(value_type[i] == 1):
					value_type[i] = 'I'
				if(value_type[i] == 2):
					value_type[i] = 'B'

			

			for i in range(m):
				k.append(0)
				site.append([])
				value.append([])
				constraint_value.append(now_conFeature[i][0].numpy().item()) 
				constraint_type.append(now_conFeature[i][1].numpy().item()) 

			for i in range(nonzero[now_num]):
				now_cons = now_edgIndices[0][i].numpy().item() - n
				now_vars = now_edgIndices[1][i].numpy().item()

				now_value = now_edgFeatures[i][0].numpy().item()

				k[now_cons] += 1
				site[now_cons].append(now_vars)
				value[now_cons].append(now_value)
			
			constr_flag = []
			new_constraint_type = []
			for i in range(m):
				LHS = 0
				for j in range(k[i]):
					LHS += value[i][j] * logits[site[i][j]]
				if(constraint_type[i] == 0):
					new_constraint_type.append(1)
					if(LHS > constraint_value[i]):
						feasible_flag = 0
						constr_flag.append(1)
					else:
						constr_flag.append(0)
					
				elif(constraint_type[i] == 2):
					new_constraint_type.append(2)
					if(LHS < constraint_value[i]):
						feasible_flag = 0
						constr_flag.append(1)
					else:
						constr_flag.append(0)
				else:
					new_constraint_type.append(3)
					if(LHS != constraint_value[i]):
						feasible_flag = 0
						constr_flag.append(1)
					else:
						constr_flag.append(0)
			
			if(feasible_flag == 0):
				choose = []
				for i in range(n):
					choose.append(0)
				for i in range(m):
					if(constr_flag[i] == 1):
						for j in range(k[i]):
							choose[site[i][j]] = 1
				print("Choose Rate:", sum(choose) / n)
				flag, sol, obj = Gurobi_solver(n, m, k, site, value, constraint_value, new_constraint_type, coefficient, time_limit - (time.time() - turn_time), obj_type[now_num], logits, choose, lower_bound, upper_bound, value_type)
				
				if(flag == 1):
					result.append(obj)
					repair_time.append(1)
				else:
					constr_flag = []
					for i in range(m):
						for j in range(k[i]):
							if(choose[site[i][j]]):
								constr_flag.append(1)
								break
						if(len(constr_flag) == i):
							constr_flag.append(0)
					choose = []
					for i in range(n):
						choose.append(0)
					for i in range(m):
						if(constr_flag[i] == 1):
							for j in range(k[i]):
								choose[site[i][j]] = 1
					print("Choose Rate:", sum(choose) / n)
					flag, sol, obj = Gurobi_solver(n, m, k, site, value, constraint_value, new_constraint_type, coefficient, time_limit - (time.time() - turn_time), obj_type[now_num], logits, choose, lower_bound, upper_bound, value_type)
					repair_time.append(2)
					if(flag == 1):
						result.append(obj)
					else:
						result.append("Infeasible!")
				
			else:
				obj = 0
				for i in range(n):
					obj += coefficient[i] * logits[i]
				result.append(obj)
				repair_time.append(0)

	log(problem_name)
	log(instance_names)
	log(result)
	log(repair_time)
	print(result)
	print(repair_time)
	
	

