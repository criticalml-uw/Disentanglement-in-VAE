import torch
random_seed = 1000
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from vae_quant_mpi_inc import VAE , setup_data_loaders
import numpy as np
from datetime import datetime
'''from torch.utils.tensorboard import SummaryWriter'''
import lib.dist as dist
from lib.flows import FactorialNormalizingFlow
import os
from matrix import Matrix
from Experiment2 import checking_orthogonality 

torch.autograd.set_detect_anomaly(True)
class CustomTensorDataset(Dataset):
    def __init__(self, epsilon_error_tensor, mu_tensor, zs_tensor):
        self.epsilon_error_tensor = epsilon_error_tensor
        self.mu_tensor = mu_tensor
        self.zs_tensor = zs_tensor

    def __len__(self):
        return self.epsilon_error_tensor.size(0)

    def __getitem__(self, index):
        epet = self.epsilon_error_tensor[index]
        mut = self.mu_tensor[index]
        zst = self.zs_tensor[index]
        return epet, mut, zst

'''class Matrix(nn.Module):
    def __init__(self, input_dimension, output_dimension,intermediate_dimension, non_linearity = False, use_cuda = False, channel_dimension = 1):
        super(Matrix,self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.intermediate_dimension = intermediate_dimension
        self.non_linearity = non_linearity
        self.channel_dimension = channel_dimension
        #self.net = nn.Sequential(
            #nn.Linear(input_dimension, 4096*1),
        #)
        #self.net = nn.Linear(input_dimension, 4096*1)
        self.net = nn.Linear(input_dimension, output_dimension)
        self.nonlin = nn.Tanh()
        if use_cuda:
            self.cuda()

    def forward(self, x):
        x = self.net(x)
        if(self.non_linearity == True):
            x_n = self.nonlin(x)
            #ep_img = x_n.view(x.size(0), 1, 64, 64)
            #ep_img_lin = x.view(x.size(0), 1, 64, 64)
            ep_img = x_n.view(x.size(0), self.channel_dimension, 64, 64)
            ep_img_lin = x.view(x.size(0), self.channel_dimension, 64, 64)
            return ep_img, ep_img_lin
        else:
            #ep_img = x.view(x.size(0), 1, 64, 64)
            ep_img = x.view(x.size(0), self.channel_dimension, 64, 64)
            return ep_img, ep_img'''

def train_one_epoch(vae, epoch_index, training_loader, optimizer, loss_fn, model, args, passing, exp, channel_dimension = 1):
    print("TRAIN ONE EPOCH")
    running_loss = 0.
    last_loss = 0.
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        epet, mut, zst = data
        '''print(len(data))
        print(len(epet))'''
        '''inputs, labels = data'''
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        epsilon_outputs, _ = model(epet)
        labels = vae.decoder.forward(zst).view(zst.size(0), channel_dimension, 64, 64)
        mu_outputs = vae.decoder.forward(mut).view(mut.size(0), channel_dimension, 64, 64)
        outputs = mu_outputs + epsilon_outputs
        #print(loss_fn(outputs, labels))
        #print(loss_or(model))
        # Compute the loss and its gradients
        loss = load_loss(loss_fn, outputs, labels,  passing, model, epet, exp, channel_dimension)
        #print("LOSS", loss.item())
        loss.backward(retain_graph=True)
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 4 == 3:
            last_loss = running_loss / 4 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            '''tb_writer.add_scalar('Loss/train', last_loss, tb_x)'''
            running_loss = 0.
    print("LAST LOSS", last_loss)
    return last_loss


def train_model(args,trainloader,optimizer,loss_fn,testloader, model, vae, nl, passing, arch, exp, channel_dimension = 1):
    

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    '''writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))'''
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(vae = vae, epoch_index = epoch_number, training_loader=trainloader, optimizer=optimizer, loss_fn=loss_fn, model = model, args = args, passing = passing, exp = exp, channel_dimension = channel_dimension)
    
        # We don't need gradients on to do reporting
        model.train(False)
    
        running_vloss = 0.0
        running_act_stoch_error = 0.0
        running_proposed_stoch_loss_accuracy = 0.0
        running_jac_approx_acc = 0.0
        for i, vdata in enumerate(testloader):
            
            vepet, vmut, vzst = vdata
            '''print("VEPET SHAPE", vepet.shape)
            print("VMUT SHAPE", vmut.shape)
            print("VMUT 0 SHAPE", vmut[0].unsqueeze(0).shape)
            print("MAX VALUE AFTER DIFFERENCE", torch.max(torch.abs(vmut[0] - vmut[4])))'''
            vepsilon_outputs, _ = model(vepet)
            #vlabels = vae.decoder.forward(vzst).view(vzst.size(0), 1, 64, 64)
            vlabels = vae.decoder.forward(vzst).view(vzst.size(0), channel_dimension, 64, 64)
            '''print("VAE DECODER JACOBIAN SHAPE", torch.autograd.functional.jacobian(vae.decoder, vmut[0].unsqueeze(0)).shape)
            print("AFTER MULTIPLICATION", torch.matmul(torch.autograd.functional.jacobian(vae.decoder, vmut[0].unsqueeze(0)), vepet.transpose(0,1)).view(vzst.size(0), 1, 64, 64).shape)'''
            #vmu_outputs = vae.decoder.forward(vmut).view(vmut.size(0), 1, 64, 64)
            vmu_outputs = vae.decoder.forward(vmut).view(vmut.size(0), channel_dimension, 64, 64)
            voutputs = vmu_outputs + vepsilon_outputs
            '''print("BEFOREEEEEEEE VMU_OUTPUTS", vmu_outputs)
            print("BEFOREEEEEEEE VOUTPUTS", voutputs)
            print("BEFOREEEEEEEE vepsilon_outputs", vepsilon_outputs)'''
            # Compute the loss and its gradients
            #print("Maximum value of MU", torch.max(vmu_outputs),"EPSILON Output through our network", torch.max(vepsilon_outputs),"OUTPUT = MU +EPSILON",  torch.max(voutputs), "LABELS",torch.max(vlabels))
            vloss = loss_fn(voutputs, vlabels)
            act_stoch_error = torch.norm(torch.sub(vlabels, vmu_outputs))
            act_stoch_error = loss_fn(vlabels, vmu_outputs)
            #vloss = torch.norm(torch.sub(voutputs, vlabels))
            '''print("######################################### shape act", act_stoch_error.shape, "ours", vloss.shape)
            print("Diff between voutputs and vlabels maximum", torch.max(torch.abs(torch.sub(voutputs, vlabels))))
            print("vloss =", vloss)
            print("Diff between jac_error and the actual error max", torch.max(torch.abs(torch.sub(act_stoch_error, vloss)))) '''
            prop_stoch_loss_accuracy = loss_fn(act_stoch_error, vloss)
            running_vloss += vloss
            running_act_stoch_error += act_stoch_error
            running_proposed_stoch_loss_accuracy += prop_stoch_loss_accuracy

            if exp == "12" and nl == "True" and epoch == (args.epochs-1):
                #print("##########################", nl)
                jac_stoch_error = jacobian_approximation_accuracy(vae, vmut, vepet,vlabels, vmu_outputs, channel_dimension=channel_dimension, loss_fn = loss_fn, vzst = vzst)
                #print("Diff between jac_error and the actual error max", torch.max(torch.abs(torch.sub(act_stoch_error, jac_stoch_error))))
                #our_approx_loss = loss_fn(act_stoch_error, vloss)
                #print("######################################### shape act", act_stoch_error.shape, "jac", jac_stoch_error.shape, "ours", our_approx_loss.shape)
                #print("######################################### maxx act", torch.max(act_stoch_error), "jac",torch.max(jac_stoch_error), "ours", torch.max(our_approx_loss))
                jac_approx_acc = loss_fn(act_stoch_error, jac_stoch_error)
                print("jacobian_approx_accuracy =", jac_approx_acc, "our_approx_loss =", prop_stoch_loss_accuracy)
                running_jac_approx_acc += jac_approx_acc
    
        avg_vloss = running_vloss / (i + 1)
        avg_act_stoch_error = running_act_stoch_error/(i+1)
        avg_proposed_stoch_loss_accuracy = running_proposed_stoch_loss_accuracy/(i+1)
        #print(exp == "12" and nl == "True" and epoch == (args.epochs-1))
        if exp == "12" and nl == "True" and epoch == (args.epochs-1):
            avg_jac_approx_acc = running_jac_approx_acc / (i + 1)
            #print('LOSS train {} valid {}, act_stoch_error {}, proposed_stoch_loss_accuracy {}, jacobian_approx_accuracy {}'.format(avg_loss, avg_vloss, avg_act_stoch_error, avg_proposed_stoch_loss_accuracy,avg_jac_approx_acc ))
        #else:
            #print('LOSS train {} valid {}, act_stoch_error {}, proposed_stoch_loss_accuracy {}'.format(avg_loss, avg_vloss, avg_act_stoch_error, avg_proposed_stoch_loss_accuracy))
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        epoch_number += 1


    save_path = args.save + exp + "/" + arch + "/"
    print("SAVEEEE PATH", save_path)
    model_path = 'model_{}_{}_{}'.format(timestamp, epoch_number, nl)
    filename = os.path.join(save_path, model_path)
    torch.save(model.state_dict(), filename)
    if exp == "12" and nl == "True":
        return avg_proposed_stoch_loss_accuracy, avg_jac_approx_acc
    elif exp == "4" and nl == "True":
        return avg_proposed_stoch_loss_accuracy, filename
    else:
        return avg_proposed_stoch_loss_accuracy

def loss_or(epsilon_decoder):
    linear_epsilon_decoder = torch.Tensor(list(epsilon_decoder.named_parameters())[0][1].cpu()).cuda()
    U, S, Vh = torch.linalg.svd(linear_epsilon_decoder)
    #print(U)
    tr = linear_epsilon_decoder.transpose(0,1)
    mm = torch.matmul(tr, linear_epsilon_decoder)
    a = torch.inverse(mm.data)
    a[a < 0] = 0
    #print(a)
    nearest_ortho = torch.matmul(linear_epsilon_decoder, torch.pow(a,0.5))
    #print(S.shape, U.shape, Vh.shape)
    V_true = torch.eye(Vh.size(0), Vh.size(1)).cuda()
    U_true, S_true, _ = torch.linalg.svd(nearest_ortho)
    loss = (torch.norm(V_true - Vh) + torch.norm(S_true - S) + torch.norm(U_true - U))/3
    return loss

def loss_or2(epsilon_decoder, epet, channel_dimension = 1):
    linear_epsilon_decoder = torch.Tensor(list(epsilon_decoder.named_parameters())[0][1].cpu()).cuda()
    _, lin_matrix_output = epsilon_decoder(epet)
    tr = linear_epsilon_decoder.transpose(0,1)
    mm = torch.matmul(tr, linear_epsilon_decoder)
    a = torch.inverse(mm.data)
    a[a < 0] = 0
    nearest_ortho = torch.matmul(linear_epsilon_decoder, torch.pow(a,0.5))
    #loss = torch.norm(lin_matrix_output - torch.matmul(nearest_ortho, epet.transpose(0,1)).view(lin_matrix_output.size(0), 1,64,64))
    loss = torch.norm(lin_matrix_output - torch.matmul(nearest_ortho, epet.transpose(0,1)).view(lin_matrix_output.size(0), channel_dimension,64,64))
    return loss


def create_local_dataset(args, z_params, q_dist, test = False):
    epsilon_error_tensor = torch.Tensor().cuda()
    mu_tensor = torch.Tensor().cuda()
    zs_tensor = torch.Tensor().cuda()
    if test:
        samples_per_data = args.samples_per_data_test
    else:
        samples_per_data = args.samples_per_data_train

    for j in range(samples_per_data):
            zs, std_z, logsigma, mu = q_dist.sample_return_everything(params=z_params)
            #print("STD_Z", torch.max(std_z),"MIN OF ABS VALUE", torch.min(torch.abs(std_z)), "LOGSIGMA", torch.max(logsigma), "EXP LOGSIGMA", torch.max(torch.exp(logsigma)))
            epsilon_error = std_z * torch.exp(logsigma)
            #print("MU" , torch.max(mu), "EPSILON_ERROR", torch.max(epsilon_error))
            input_dimension = epsilon_error.size(1)
            #output_dimension = epsilon_error.size(1)
            epsilon_error_tensor = torch.cat((epsilon_error_tensor, epsilon_error), 0)
            mu_tensor = torch.cat((mu_tensor, mu), 0)
            zs_tensor = torch.cat((zs_tensor, zs), 0)
    return mu_tensor, zs_tensor, epsilon_error_tensor, input_dimension

def jacobian_approximation_accuracy(vae, vmut, vepet,vlabels, vmu_outputs, channel_dimension = 1, loss_fn = None, vzst=None):
    #act_stoch_error = torch.norm(torch.sub(vlabels, vmu_outputs))
    #print("Original stochastic error, difference between vlabels and mu(z)", torch.max(torch.abs(torch.sub(vlabels, vmu_outputs))))
    #act_stoch_error = loss_fn(vlabels, vmu_outputs)
    jac = torch.autograd.functional.jacobian(vae.decoder, vzst[0].unsqueeze(0))
    jac_stoch_error = torch.norm(torch.matmul(jac, vepet.transpose(0,1)).view(vepet.size(0), channel_dimension, 64, 64))
    #jac_stoch_error = torch.square(torch.matmul(jac, vepet.transpose(0,1)).view(vepet.size(0), channel_dimension, 64, 64))
    #return act_stoch_error, jac_stoch_error
    return jac_stoch_error

def load_loss(loss_fn, outputs, labels,  passing, model, epet, exp, channel_dimension):
    if exp == "12":
        loss = loss_fn(outputs, labels)
    else:
        if passing == 1:
            loss = loss_fn(outputs, labels) + loss_or2(model, epet, channel_dimension)
        elif  passing == 2:
            loss = loss_fn(outputs, labels) - 0.7*loss_or2(model, epet, channel_dimension)
        else:
            loss = loss_fn(outputs, labels) - 1.4*loss_or2(model, epet, channel_dimension)
    return loss

def checking_nonlinearity_efficacy(args):

    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()
    if args.dataset == 'mnist':
        mnist = True
    else:
        mnist = False

    if args.dataset == '3dshapes':
        d3 = True
    else:
        d3 = False
    if args.dataset == 'mpi3D':
        mpi = True
        print("MPI", mpi)
    else:
        mpi = False
    architectures = ["beta_tcvae", "beta_vae", "vae"]
    '''vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, mnist = mnist, d3 = d3)
    checkpoint = torch.load(args.path)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()'''
    '''for arch, path in zip(architectures, args.paths):
        vae = VAE(z_dim=args.latent_dim, x_dist = dist.Normal(), use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
                include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, mnist = mnist, d3 = d3, mpi = mpi)
        checkpoint = torch.load(path)
        vae.load_state_dict(checkpoint['state_dict'])'''
    train_loader = setup_data_loaders(args, use_cuda=True)
    example_samples_evaluate = np.random.randint(low = 0, high = len(train_loader.dataset), size=(args.size_train))
    #experiments = ["12", "4"]
    experiments = ["4", "12"]
    sum_ortho_list = [0.0,0.0,0.0]
    for exp in experiments:
        print ("EXP", exp, "STARTS")
        for i in example_samples_evaluate:
            file_path_list = []
            passing = 1
            for arch, path in zip(architectures, args.paths):
                print("############################################################ARCHITECTURE", arch, path)
                vae = VAE(z_dim=args.latent_dim, x_dist = dist.Normal(), use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
                include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, mnist = mnist, d3 = d3, mpi = mpi)
                checkpoint = torch.load(path)
                vae.load_state_dict(checkpoint['state_dict'])
                vae.eval()
                data = train_loader.dataset[i]
                data = data.unsqueeze(0)
                '''if args.dataset == 'shapes':'''
                data = data.view(data.size(0), data.size(1), 64, 64).cuda()
                output_dimension = data.size(1)*data.size(2)*data.size(3)
                channel_dimension = data.size(1)
                print("DATA SHAPE", data.shape, channel_dimension, output_dimension)
                '''else:
                    data = data.view(data.size(0), 1, 64, 64).cuda()'''
                z_params = vae.encoder.forward(data).view(data.size(0), args.latent_dim, q_dist.nparams).cuda() 
                mu_tensor_train, zs_tensor_train, epsilon_error_tensor_train, input_dimension = create_local_dataset(args, z_params, q_dist)
                train_dataset = CustomTensorDataset(epsilon_error_tensor=epsilon_error_tensor_train, mu_tensor=mu_tensor_train, zs_tensor= zs_tensor_train)
                trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size_exp1, shuffle=True)
                mu_tensor_test, zs_tensor_test, epsilon_error_tensor_test, input_dimension = create_local_dataset(args, z_params, q_dist)
                test_dataset = CustomTensorDataset(epsilon_error_tensor=epsilon_error_tensor_test, mu_tensor=mu_tensor_test, zs_tensor= zs_tensor_test)
                testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size_exp1, shuffle=True)
                print("LENGTH OF TRAIN DATASET", len(trainloader.dataset))
                print("LENGTH OF TEST DATASET", len(testloader.dataset))
                print("INPUT DIMENSION", input_dimension)
                print("OUTPUT DIMENSION", output_dimension)
                if exp == "12":
                    model1 = Matrix(input_dimension, output_dimension,intermediate_dimension = args.intermediate_dimension, non_linearity = True, use_cuda = True, channel_dimension = channel_dimension)
                    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1, momentum=0.9)
                    loss_fn1 = nn.MSELoss()
                    avg_proposed_stoch_loss_accuracy_non_linear, avg_jac_approx_acc = train_model(args=args, trainloader=trainloader, optimizer=optimizer1, loss_fn=loss_fn1, testloader=testloader , model = model1, vae = vae, nl = "True", passing = passing, arch = arch, exp=exp, channel_dimension = channel_dimension)
                    print("AVG_VLOSS", avg_proposed_stoch_loss_accuracy_non_linear, "AVG JACOBIAN VLOSS", avg_jac_approx_acc)
                else:
                    model3 = Matrix(input_dimension, output_dimension,intermediate_dimension = args.intermediate_dimension, non_linearity = True, use_cuda = True, channel_dimension = channel_dimension)
                    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.1, momentum=0.9)
                    loss_fn3 = nn.MSELoss()
                    avg_vloss_non_linear, file_path = train_model(args=args, trainloader=trainloader, optimizer=optimizer3, loss_fn=loss_fn3, testloader=testloader , model = model3, vae = vae, nl = "True", passing = passing, arch = arch, exp=exp, channel_dimension = channel_dimension)
                    file_path_list.append(file_path)
                if exp == "12":
                    model2 = Matrix(input_dimension, output_dimension,intermediate_dimension = args.intermediate_dimension, non_linearity = False, use_cuda = True, channel_dimension = channel_dimension)
                    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.0001, momentum=0.9)
                    loss_fn2 = nn.MSELoss()
                    avg_proposed_stoch_loss_accuracy_linear = train_model(args=args, trainloader=trainloader, optimizer=optimizer2, loss_fn=loss_fn2, testloader=testloader, model = model2, vae = vae, nl = "False", passing = passing,arch = arch, exp=exp, channel_dimension = channel_dimension)
                    #print("AVG_VLOSS NON LINEAR", avg_vloss_non_linear, "AVG VLOSS LINEAR", avg_vloss_linear)
                    print("AVG VLOSS LINEAR", avg_proposed_stoch_loss_accuracy_linear, "AVG_VLOSS", avg_proposed_stoch_loss_accuracy_non_linear)
                else:
                    passing += 1
            if exp == "4":
                orthogonality_list = checking_orthogonality(file_path_list, args.dataset, False, channel_dimension, intermediate_dimension = args.intermediate_dimension)
                print("ORTHOGONALITY LIST", orthogonality_list)
                print("LEN ORTHO LIST", len(orthogonality_list), "LEN SUM LIST", len(sum_ortho_list))
                sum_ortho_list = [sum_ortho_list[i] + orthogonality_list[i] for i in range(len(orthogonality_list))]
        if exp == "4":
            avg_ortho_list = [sum_ortho_list_ele/args.size_train for sum_ortho_list_ele in sum_ortho_list]
            print(avg_ortho_list)

def list_of_strings(arg):
    return arg.split(',')

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name', choices=['shapes', 'faces', '3dshapes', "mpi3D"])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--save', default='Just_Exp/Seed1234/')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    parser.add_argument('-intr', '--intermediate_dimension', default = 100, type = int)
    '''parser.add_argument('-p', '--path', type = str)'''
    parser.add_argument('-p', '--paths', type = list_of_strings)
    parser.add_argument('-ba', '--batch_size_exp1', default = 32, type = int)
    parser.add_argument('-st', '--size_train', default = 1, type = int)
    parser.add_argument('-spdtr', '--samples_per_data_train', default = 10, type = int)
    parser.add_argument('-spdte', '--samples_per_data_test', default = 16, type = int)
    parser.add_argument('-e', '--epochs', default=1, type = int)
    args = parser.parse_args()
    print(args.paths)
    torch.cuda.set_device(args.gpu)

    checking_nonlinearity_efficacy(args)



if __name__ == '__main__':
    model = main()






    
