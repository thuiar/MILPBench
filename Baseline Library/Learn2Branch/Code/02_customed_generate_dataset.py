import os
import glob
import gzip
import argparse
import pickle
import queue
import shutil
import threading
import numpy as np
import ecole
from collections import namedtuple


class ExploreThenStrongBranch:
    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()  
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        probabilities = [1-self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model,done), True)
        else:
            return (self.pseudocosts_function.extract(model,done), False)


def send_orders(orders_queue, instances, seed, query_expert_prob, time_limit, out_dir, stop_flag):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : queue.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    while not stop_flag.is_set():
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, query_expert_prob, time_limit, out_dir])
        episode += 1


def make_samples(in_queue, out_queue, stop_flag):
    """
    Worker loop: fetch an instance, run an episode and record samples.
    Parameters
    ----------
    in_queue : queue.Queue
        Input queue from which orders are received.
    out_queue : queue.Queue
        Output queue in which to send samples.
    stop_flag: threading.Event
        A flag to tell the thread to stop.
    """
    sample_counter = 0
    while not stop_flag.is_set():
        episode, instance, seed, query_expert_prob, time_limit, out_dir = in_queue.get()

        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0,
                           'limits/time': time_limit, 'timing/clocktype': 2}
        observation_function = { "scores": ExploreThenStrongBranch(expert_probability=query_expert_prob),
                                 "node_observation": ecole.observation.NodeBipartite() }
        env = ecole.environment.Branching(observation_function=observation_function,
                                          scip_params=scip_parameters, pseudo_candidates=True)

        print(f"[w {threading.current_thread().name}] episode {episode}, seed {seed}, "
              f"processing instance '{instance}'...\n", end='')
        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        env.seed(seed)
        observation, action_set, _, done, _ = env.reset(instance)
        while not done:
            scores, scores_are_expert = observation["scores"]
            node_observation = observation["node_observation"]
            node_observation = (node_observation.row_features,
                                (node_observation.edge_features.indices,
                                 node_observation.edge_features.values),
                                node_observation.variable_features)

            action = action_set[scores[action_set].argmax()]

            if scores_are_expert and not stop_flag.is_set():
                data = [node_observation, action, action_set, scores]
                filename = f'{out_dir}/sample_{episode}_{sample_counter}.pkl'

                with gzip.open(filename, 'wb') as f:
                    pickle.dump({
                        'episode': episode,
                        'instance': instance,
                        'seed': seed,
                        'data': data,
                        }, f)
                out_queue.put({
                    'type': 'sample',
                    'episode': episode,
                    'instance': instance,
                    'seed': seed,
                    'filename': filename,
                })
                sample_counter += 1

            try:
                observation, action_set, _, done, _ = env.step(action)
            except Exception as e:
                done = True
                with open("error_log.txt","a") as f:
                    f.write(f"Error occurred solving {instance} with seed {seed}\n")
                    f.write(f"{e}\n")

        print(f"[w {threading.current_thread().name}] episode {episode} done, {sample_counter} samples\n", end='')
        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    query_expert_prob, time_limit):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.
    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = queue.Queue(maxsize=2*n_jobs)
    answers_queue = queue.SimpleQueue()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher_stop_flag = threading.Event()
    dispatcher = threading.Thread(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), query_expert_prob,
                  time_limit, tmp_samples_dir, dispatcher_stop_flag),
            daemon=True)
    dispatcher.start()

    workers = []
    workers_stop_flag = threading.Event()
    for i in range(n_jobs):
        p = threading.Thread(
                target=make_samples,
                args=(orders_queue, answers_queue, workers_stop_flag),
                daemon=True)
        workers.append(p)
        p.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {threading.current_thread().name}] {i} / {n_samples} samples written, "
                          f"ep {sample['episode']} ({in_buffer} in buffer).\n", end='')

                    # early stop dispatcher
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher_stop_flag.set()
                        print(f"[m {threading.current_thread().name}] dispatcher stopped...\n", end='')

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # # stop all workers
    workers_stop_flag.set()
    for p in workers:
        p.join()

    print(f"Done collecting samples for {out_dir}")
    shutil.rmtree(tmp_samples_dir, ignore_errors=True)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=[
                 'Cut',
                 'MIPlib',
                 'miplib_mixed_neos',
                 'Nexp',
                 'Transportation',
                 'vary_bounds_s1',
                 'vary_bounds_s2',
                 'vary_bounds_s3',
                 'vary_matrix_rhs_bounds_obj_s1',
                 'vary_matrix_s1',
                 'vary_obj_s1',
                 'vary_obj_s2',
                 'vary_obj_s3',
                 'vary_rhs_obj_s2',
                 'vary_rhs_s2',
                 'vary_rhs_s4',
                 'knapsack',
                 'mis',
                 'setcover',
                 'corlat',
                 '1_item_placement',
                 '2_load_balancing',
                 'MVC_easy_instance'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    print(f"seed {args.seed}")

    node_record_prob = 0.05
    time_limit = 3600

    if args.problem == None:
        raise NotImplementedError
    else:
        instances_train = glob.glob(f'data/instances/{args.problem}/train/*.lp')
        instances_valid = glob.glob(f'data/instances/{args.problem}/valid/*.lp')
        instances_test = glob.glob(f'data/instances/{args.problem}/test/*.lp')
        train_size = len(instances_train)*10
        valid_size = len(instances_valid)*10
        test_size = len(instances_test)*10
        out_dir = f'data/samples/{args.problem}'

    if args.problem == 'miplib_mixed_neos':
        train_size = len(instances_train)*5
        valid_size = len(instances_valid)*5
        test_size = len(instances_test)*5

    if args.problem == 'vary_bounds_s2' or args.problem == 'vary_bounds_s3':
        train_size = len(instances_train)
        valid_size = len(instances_valid)
        test_size = len(instances_test)

    # if args.problem == 'setcover':
    #     instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
    #     instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
    #     instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
    #     train_size = len(instances_train)*100
    #     valid_size = len(instances_valid)*100
    #     test_size = len(instances_test)*100
    #     out_dir = 'data/samples/setcover/500r_1000c_0.05d'

    # elif args.problem == 'vary_bounds_s1':
    #     instances_train = glob.glob('data/instances/vary_bounds_s1/train/*.lp')
    #     instances_valid = glob.glob('data/instances/vary_bounds_s1/valid/*.lp')
    #     instances_test = glob.glob('data/instances/vary_bounds_s1/test/*.lp')
    #     train_size = len(instances_train)*100
    #     valid_size = len(instances_valid)*100
    #     test_size = len(instances_test)*100
    #     out_dir = 'data/samples/vary_bounds_s1'

    # elif args.problem == 'custom':
    #     instances_train = glob.glob('data/instances/custom/train/*.lp')
    #     instances_valid = glob.glob('data/instances/custom/valid/*.lp')
    #     instances_test = glob.glob('data/instances/custom/test/*.lp')
    #     train_size = len(instances_train)*100
    #     valid_size = len(instances_valid)*100
    #     test_size = len(instances_test)*100
    #     out_dir = 'data/samples/custom'


        

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    # create output directory, throws an error if it already exists
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, valid_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit)

    rng = np.random.RandomState(args.seed + 2)
    collect_samples(instances_test, out_dir + '/test', rng, test_size,
                    args.njobs, query_expert_prob=node_record_prob,
                    time_limit=time_limit)
