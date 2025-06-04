# tensorflow=2.4
import numpy as np
from pandas import read_csv
import tensorflow as tf
import argparse
import os
from models import GCNPolicy

## ARGUMENTS OF THE SCRIPT
# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu", 		help="gpu index", 					default="0")
# parser.add_argument("--embSize", 	help="embedding size of GNN", 		default="6")
# parser.add_argument("--epoch", 		help="num of epoch", 				default="10000")
# parser.add_argument("--type", 		help="what's the type of the model",default="fea", 	choices = ['fea','obj','sol'])
# parser.add_argument("--data_path", 	help="path of data", 				default=None)
# args = parser.parse_args()

## FUNCTION OF TRAINING PER EPOCH
def process(model, dataloader, optimizer, type = 'fea'):
	#num = random.randint(0, len(n_csm) - 1)
	#conFeatures, edgIndices, edgFeatures, varFeatures, n_Cons.sum(), n_Vars.sum(), n_Cons, n_Vars, nonzero, labels
	conFeature_list, edgIndices_list, edgFeatures_list, varFeatures_list, n_cs, n_vs, n_Cons, n_Vars, nonzero, cand_scores_list = dataloader
	return_loss = 0
	for i in range(len(n_Cons)):
		#print("=====", i)
		batched_states = (conFeature_list[i], edgIndices_list[i], edgFeatures_list[i], varFeatures_list[i], n_Cons[i], n_Vars[i], n_Cons[i], n_Vars[i])  
		with tf.GradientTape() as tape:
			logits = model(batched_states, tf.convert_to_tensor(True)) 
			loss = tf.keras.metrics.mean_squared_error(cand_scores_list[i], logits)
			loss = tf.reduce_mean(loss)
		return_loss += loss
		
	grads = tape.gradient(target=loss, sources=model.variables)
	optimizer.apply_gradients(zip(grads, model.variables))

	errs = None
	err_rate = None
	

	return return_loss.numpy(), errs, err_rate


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

	## SET-UP HYPER PARAMETERS
	max_epochs = 200
	lr = 0.003
	seed = 0

	## SET-UP DATASET
	trainfolder = f'./instances/{problem_name}/train/'
	gpu_index = 1
	embSize = 6

	## SET-UP MODEL
	if not os.path.exists('./saved-models/'):
		os.mkdir('./saved-models/')
	model_setting = trainfolder.replace('/','-')
	model_path = './saved-models/' + model_setting + '-' + 'obj' + '-s' + str(embSize) + '.pkl'

	## LOAD DATASET INTO MEMORY

	vars_all = 0
	cons_all = 0


	varFeatures = read_csv(trainfolder + "/VarFeatures_feas.csv", header=None)
	conFeatures = read_csv(trainfolder + "/ConFeatures_feas.csv", header=None)
	edgFeatures = read_csv(trainfolder + "/EdgeFeatures_feas.csv", header=None)
	edgIndices = read_csv(trainfolder + "/EdgeIndices_feas.csv", header=None)
	labels = read_csv(trainfolder + "/Labels_solu.csv", header=None)
	n_Cons = read_csv(trainfolder + "/Con_num.csv", header=None).values.reshape(-1)
	n_Vars = read_csv(trainfolder + "/Var_num.csv", header=None).values.reshape(-1)
	nonzero = read_csv(trainfolder + "/Nonzero_num.csv", header=None).values.reshape(-1)

	nConsF = conFeatures.shape[1]
	nVarF = varFeatures.shape[1]
	nEdgeF = edgFeatures.shape[1]


	## SET-UP TENSORFLOW
	tf.random.set_seed(seed)
	gpu_index = int(gpu_index)
	tf.config.set_soft_device_placement(True)
	gpus = tf.config.list_physical_devices('GPU')
	tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
	tf.config.experimental.set_memory_growth(gpus[gpu_index], True)

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
		for i in range(len(n_Cons)):
			varFeatures_list.append(tf.constant(varFeatures.values[now_n : now_n + n_Vars[i], :], dtype=tf.float32))
			conFeature_list.append(tf.constant(conFeatures.values[now_m : now_m + n_Cons[i], :], dtype=tf.float32))
			edgFeatures_list.append(tf.constant(edgFeatures.values[now_nonzero : now_nonzero + nonzero[i], :], dtype=tf.float32))
			edgIndices_list.append(tf.transpose(tf.constant(edgIndices.values[now_nonzero : now_nonzero + nonzero[i], :], dtype=tf.int32)))
			labels_list.append(tf.constant(labels.values[now_n : now_n + n_Vars[i], :], dtype=tf.float32))
			now_n += n_Vars[i]
			now_m += n_Cons[i]
			now_nonzero += nonzero[i]

		train_data = (conFeature_list, edgIndices_list, edgFeatures_list, varFeatures_list, n_Cons.sum(), n_Vars.sum(), n_Cons, n_Vars, nonzero, labels_list)

		### INITIALIZATION ###
		model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
		loss_init,_,_ = process(model, train_data, optimizer, type ='sol')

		epoch = 0
		count_restart = 0
		err_best = 2
		loss_best = 1e10
		
		### MAIN LOOP ###
		while epoch <= max_epochs:
			train_loss,errs,err_rate = process(model, train_data, optimizer, type = 'sol')
			print("Loss", epoch, train_loss)
				
			print(f"EPOCH: {epoch}, TRAIN LOSS: {train_loss}")
			if train_loss < loss_best:
				model.save_state(model_path)
				print("model saved to:", model_path)
				loss_best = train_loss
			
			## If the loss does not go down, we restart the training to re-try another initialization.
			if epoch == 200 and count_restart < 3 and (train_loss > loss_init * 0.8 or (err_rate != None and err_rate > 0.5)):
				print("Fail to reduce loss, restart...")
				model = GCNPolicy(embSize, nConsF, nEdgeF, nVarF, isGraphLevel = False)
				optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
				loss_init,_,_ = process(model, train_data, optimizer, type = 'sol')
				epoch = 0
				count_restart += 1
				
			epoch += 1
		
		print("Count of restart:", count_restart)
		model.summary()
	
	

