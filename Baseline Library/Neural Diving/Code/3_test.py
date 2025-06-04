import argparse
import pickle
from pathlib import Path
from typing import Union
from gurobipy import *

import os
import time
import torch
import torch.nn.functional as F
import torch_geometric
from pytorch_metric_learning import losses

from model.graphcnn import GNNPolicy

__all__ = ["train"]

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

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        assignment
    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.assignment = assignment

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with open(self.sample_files[index], "rb") as f:
            [variable_features, constraint_features, edge_indices, edge_features, solution] = pickle.load(f)

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(variable_features),
            torch.FloatTensor(solution)
        )

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = len(constraint_features) + len(variable_features)
        graph.cons_nodes = len(constraint_features)
        graph.vars_nodes = len(variable_features)

        return graph


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output

def process(policy, data_loader, device, optimizer=None):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            #print("QwQ")
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits, select = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
            
            n = len(batch.variable_features)
            choose = {}
            for i in range(n):
                if(select[i] >= 0.5):
                    choose[i] = 0
                else:
                    choose[i] = 1
            new_idx_train = []
            for i in range(n):
                if(choose[i]):
                    new_idx_train.append(i)
            
            set_c = 0.7
            if(len(new_idx_train) < set_c * n):
                loss_select = (set_c - len(new_idx_train) / n) ** 2
            else:
                loss_select = 0
            #print(batch.constraint_features)
            #print(batch.edge_index)
            #print(batch.edge_attr)
            #print(batch.variable_features)
            # Index the results by the candidates, and split and pad them
            # logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            #loss = F.binary_cross_entropy(logits, batch.assignment)
            loss_func = torch.nn.MSELoss()
            #print(logits)
            #print(logits)
            #print(batch.assignment)
            #print(logits)
            loss = loss_func(logits[new_idx_train], batch.assignment[new_idx_train]) + loss_select
            
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            mean_loss += loss.item() * batch.num_graphs
            # mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    # mean_acc /= n_samples_processed
    return mean_loss

def log(any, txt = 'Nerual-Diving_result.txt'):
    with open(txt, 'a')as f:
        f.writelines(str(any))
        f.writelines('\n')

def test(
    path: str,
    model_path: Union[str, Path],
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    num_epochs: int = 20,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    This function trains a GNN policy on training data. 

    Args:
        data_path: Path to the data directory.
        model_save_path: Path to save the model.
        batch_size: Batch size for training.
        learning_rate: Learning rate for the optimizer.
        num_epochs: Number of epochs to train for.
        device: Device to use for training.
    """

    time_limit = 100
    if path == '1_item_placement':
        time_limit = 4000
    elif path == 'Cut':
        time_limit = 4000
    elif path == 'MIPlib':
        time_limit = 150
    elif path == 'miplib_mixed_neos':
        time_limit = 4000
    elif path == 'Nexp':
        time_limit = 4000
    elif path == 'Transportation':
        time_limit = 4000
    elif path == 'vary_bounds_s1':
        time_limit = 400
    elif path == 'vary_bounds_s2':
        time_limit = 1000
    elif path == 'vary_bounds_s3':
        time_limit = 1000
    elif path == 'vary_matrix_rhs_bounds_obj_s1':
        time_limit = 100
    elif path == 'vary_matrix_s1':
        time_limit = 100
    elif path == 'vary_obj_s1':
        time_limit = 100
    elif path == 'vary_obj_s2':
        time_limit = 150
    elif path == 'vary_obj_s3':
        time_limit = 100
    elif path == 'vary_rhs_obj_s2':
        time_limit = 100
    elif path == 'vary_rhs_s2':
        time_limit = 100
    elif path == 'vary_rhs_s2':
        time_limit = 100
    elif path == 'vary_rhs_s4':
        time_limit = 100

    model_path = f'{model_path}/{path}_trained.pkl'
    policy = GNNPolicy().to(device)
    policy.load_state_dict(torch.load(model_path, policy.state_dict()))

    for filename in os.listdir(f'instances/{path}/LP_test'):
        if filename.endswith('.lp'):
            file_path = os.path.join(lp_path, filename)
            lp_files.append(file_path)
            lp_name.append(filename)

    path = f'instances/{path}/test'
    sample_files = [str(path) for path in Path(path).glob("pair*.pickle")]
    number = len(sample_files)
    
    result = []
    for num in range(number):
        File = []
        if(os.path.exists(path + '/pair' + str(num) + '.pickle') == False):
            print("No input file!")
            return 
        File.append(path + '/pair' + str(num) + '.pickle')
        data = GraphDataset(File)
        loader = torch_geometric.loader.DataLoader(data, batch_size = 1)

        for batch in loader:
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits, select = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
            )
        #print(logits)
        
        if(os.path.exists(path + '/data' + str(num) + '.pickle') == False):
            print("No problem file!")

        with open(path + '/data' + str(num) + '.pickle', "rb") as f:
            problem = pickle.load(f)
        
        obj_type = problem[0]
        n = problem[1]
        m = problem[2]
        k = problem[3]
        site = problem[4]
        value = problem[5]
        constraint = problem[6]
        constraint_type = problem[7]
        coefficient = problem[8]
        lower_bound = problem[9]
        upper_bound = problem[10]
        value_type = problem[11]

        new_select = select.to('cpu').detach().numpy() 
        new_select.sort()

        now_sol = logits.to('cpu').detach().numpy() 
        for i in range(n):
            if(value_type[i] != 'C'):
                now_sol[i] = int(now_sol[i] + 0.5)
            now_sol[i] = min(now_sol[i], upper_bound[i])
            now_sol[i] = max(now_sol[i], lower_bound[i])

        add_flag = 0 
        for turn in range(11):
            choose = []
            rate = (int)(0.1 * turn * n)
            for i in range(n):
                if(select[i] >= new_select[rate]):
                    choose.append(1)
                else:
                    choose.append(0)
            #print(0.1 * turn, sum(choose) / n)
            flag, sol, obj = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, now_sol, choose, lower_bound, upper_bound, value_type)
            if(flag == 1):
                add_flag = 1
                result.append(obj)
                break
        if(add_flag == 0):
            result.append("infeaseible")

    log(path)
    log(instance_names)
    log(result)
    print(result)

def parse_args():
    """
    This function parses the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default='fc.data', help="Path for test Data.")
    parser.add_argument("--model_path", type=str, default="trained_model", help="Path to save the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(**vars(args))
