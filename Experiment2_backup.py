import torch
import lib.dist as dist
import argparse
from Experiment1 import Matrix
import torch.nn as nn



def checking_orthogonality(path):
    input_dimension = 10 
    output_dimension = 4096
    intermediate_dimension = 1200
    epsilon_decoder = Matrix(input_dimension, output_dimension,intermediate_dimension = intermediate_dimension, non_linearity = True, use_cuda = True)
    print(path)
    checkpoint = torch.load(path)
    '''print("CHECKPOINT", checkpoint)'''
    epsilon_decoder.load_state_dict(checkpoint)
    epsilon_decoder.eval()
    '''print(torch.Tensor(list(epsilon_decoder.named_parameters())[0][1].cpu())[0])'''
    linear_epsilon_decoder = torch.Tensor(list(epsilon_decoder.named_parameters())[0][1].cpu())
    '''pad = int((linear_epsilon_decoder.size(0) - linear_epsilon_decoder.size(1))/2)
    pad_tuple = (pad,pad)
    linear_epsilon_decoder = torch.nn.functional.pad(input=linear_epsilon_decoder, pad=pad_tuple, mode='constant', value=0)'''
    U, S, Vh = torch.linalg.svd(linear_epsilon_decoder)
    #print(linear_epsilon_decoder.shape)
    #trans = linear_epsilon_decoder.transpose(0,1)
    #print(trans.shape, linear_epsilon_decoder.shape)
    #print(torch.mul( linear_epsilon_decoder,trans).shape)
    #print("MATRIXXXX", linear_epsilon_decoder)  
    #print("MATRIXXXX TRANSPOSE", linear_epsilon_decoder.transpose(0,1))  
    #print(torch.matmul(linear_epsilon_decoder.transpose(0,1), linear_epsilon_decoder))
    a = torch.inverse(torch.matmul(linear_epsilon_decoder.transpose(0,1), linear_epsilon_decoder))
    a[a < 0] = 0
    #print(a)
    nearest_ortho = torch.matmul(linear_epsilon_decoder, torch.pow(a,0.5))
    print(S.shape, U.shape, Vh.shape)
    colm_norm = torch.norm(linear_epsilon_decoder, dim=0)
    #print("COLM NORM", colm_norm)
    '''S_true = torch.diag(colm_norm)'''
    V_true = torch.eye(Vh.size(0), Vh.size(1))
    '''U_true = torch.div(linear_epsilon_decoder, colm_norm)
    m = nn.ZeroPad2d((0, output_dimension-input_dimension, 0, 0))
    U_true = m(U_true)'''
    U_true, S_true, _ = torch.linalg.svd(nearest_ortho)
    print(colm_norm.shape,U_true.shape,V_true.shape)
    '''nearest_orthogonal_matrix = torch.matmul(U,Vh)
    print(torch.subtract(linear_epsilon_decoder, nearest_orthogonal_matrix))
    print(torch.norm(torch.subtract(linear_epsilon_decoder, nearest_orthogonal_matrix)))'''
    print("Vh Diff", torch.norm(torch.subtract(V_true,Vh)))
    #print("S DIFF", torch.norm(torch.subtract(colm_norm,S)))
    print("S DIFF", torch.norm(torch.subtract(S_true,S)))
    print("U DIFF", torch.norm(torch.subtract(U_true,U)))
    print("DIST ORTHOOO 2 REPEAT", (torch.norm(torch.subtract(V_true,Vh)) + torch.norm(torch.subtract(S_true,S)) + torch.norm(torch.subtract(U_true,U)))/3)
    loss_fn = nn.MSELoss()
    #distance_from_orthogonality = loss_fn(linear_epsilon_decoder, nearest_orthogonal_matrix)
    distance_from_orthogonality = loss_fn(Vh, V_true)
    #dist_ortho_2 = loss_fn(nearest_ortho, linear_epsilon_decoder)
    dist_ortho_2 = torch.norm(torch.subtract(linear_epsilon_decoder, nearest_ortho))
    print("DIST OTHOOOOOOOOO 2", dist_ortho_2)
    return distance_from_orthogonality

def list_of_strings(arg):
    return arg.split(',')

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-p', '--paths', type = list_of_strings)
    args = parser.parse_args()

    distance_from_orthogonality = checking_orthogonality(path = args.path)
    print(distance_from_orthogonality)



if __name__ == '__main__':
    model = main()






    
