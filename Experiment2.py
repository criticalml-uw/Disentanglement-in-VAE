import torch
import lib.dist as dist
import argparse
from matrix import Matrix
import torch.nn as nn
import numpy as np



def checking_orthogonality(paths, dataset, norm=False, channel_dimension = 3, intermediate_dimension = 1200):
    udist_list = []
    vdist_list = []
    sdist_list = []
    orthodist_list = []
    for path in paths:
        input_dimension = 10 
        output_dimension = 4096*channel_dimension
        #intermediate_dimension = 1200
        epsilon_decoder = Matrix(input_dimension, output_dimension,intermediate_dimension = intermediate_dimension, non_linearity = True, use_cuda = True)
        print(path)
        checkpoint = torch.load(path)
        epsilon_decoder.load_state_dict(checkpoint)
        epsilon_decoder.eval()
        linear_epsilon_decoder = torch.Tensor(list(epsilon_decoder.named_parameters())[0][1].cpu())
        U, S, Vh = torch.linalg.svd(linear_epsilon_decoder)
        a = torch.inverse(torch.matmul(linear_epsilon_decoder.transpose(0,1), linear_epsilon_decoder))
        a[a < 0] = 0
        nearest_ortho = torch.matmul(linear_epsilon_decoder, torch.pow(a,0.5))
        print(S.shape, U.shape, Vh.shape)
        V_true = torch.eye(Vh.size(0), Vh.size(1))
        U_true, S_true, _ = torch.linalg.svd(nearest_ortho)
        udist_list.append(torch.norm(torch.subtract(U_true,U)).tolist())
        sdist_list.append(torch.norm(torch.subtract(S_true,S)).tolist())
        vdist_list.append(torch.norm(torch.subtract(V_true,Vh)).tolist())
    if norm:
        udist_list = [x/max(udist_list) for x in udist_list]
        vdist_list = [x/max(vdist_list) for x in vdist_list]
        sdist_list = [x/max(sdist_list) for x in sdist_list]
    for udist, vdist, sdist in zip(udist_list, vdist_list, sdist_list):
        orthodist_list.append((udist + vdist + sdist)/3)
    return orthodist_list

def list_of_strings(arg):
    return arg.split(',')

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-p', '--paths', type = list_of_strings)
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name', choices=['shapes', 'faces', '3dshapes', 'mpi3D'])
    parser.add_argument('--norm', action='store_true', help='Normalize the result per metric')
    args = parser.parse_args()

    distance_from_orthogonality_list = checking_orthogonality(args.paths, args.dataset, args.norm)
    print(distance_from_orthogonality_list)



if __name__ == '__main__':
    model = main()






    
