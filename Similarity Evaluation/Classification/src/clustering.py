import torch
import numpy as np
from sklearn.cluster import SpectralClustering
import argparse
import pickle

def spectral_clustering(tensor, names, n_clusters):
    data = tensor.detach().numpy()
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=0).fit(data)
    labels = clustering.labels_
    clusters = {i: [] for i in range(n_clusters)}
    for name, label in zip(names, labels):
        clusters[label].append(name)
    
    return clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()
    with open(args.input_file, 'rb') as f:
        tensor = pickle.load(f)
        f.close()

    with open(args.filename, 'rb') as f:
        names = pickle.load(f)
        f.close()

    n_clusters = 50
    # print(names)
    
    clusters = spectral_clustering(tensor, names, n_clusters)
    
    for cluster_id, cluster_names in clusters.items():
        print(f"Cluster {cluster_id}: {cluster_names}")