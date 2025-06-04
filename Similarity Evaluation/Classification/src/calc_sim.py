import torch
import argparse
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file1', type=str)
    parser.add_argument('--input_file2', type=str)
    args = parser.parse_args()

    f1 = open(args.input_file1, 'rb')
    A = pickle.load(f1)
    ret = 0
    if args.input_file2 is not None:
        f2 = open(args.input_file2, 'rb')
        B = pickle.load(f2)
        for i in range(A.shape[0]):
            for j in range(B.shape[0]):
                ret += torch.dist(A[i, :], B[j, :])
        ret /= (A.shape[0] * B.shape[0])
        print(f"Similarity={ret}")
    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[0]):
                ret += torch.dist(A[i, :], A[j, :])
        ret /= (A.shape[0] * A.shape[0])
        print(f"Similarity={ret}")

    