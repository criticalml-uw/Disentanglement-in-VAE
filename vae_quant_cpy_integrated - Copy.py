import os
import time
import math
from numbers import Number
import argparse
import torch

import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.anneal_kl import anneal_kl
from utils.utils import plot_elbo
from utils.display_image_win import display_samples
from datasets.loading_dataset import setup_data_loaders
from models.vae import VAE


import lib.dist as dist
import utils.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow


from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401

from trainer.trainer import Trainer

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
        choices=['shapes', 'faces', 'mnist', '3dshapes', 'mpi3D'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-4, type=float, help='learning rate')
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
    parser.add_argument('--save', default='../Results/beta_tcvae/mpi')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

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
    else:
        mpi = False
    '''print("MPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", mpi)'''
    # data loader
    train_loader = setup_data_loaders(args, use_cuda=True)

    # setup the VAE
    if args.dist == 'normal':
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
        x_dist = dist.Normal()
    elif args.dist == 'laplace':
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
        x_dist = dist.Laplace()
    elif args.dist == 'flow':
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()
        x_dist = dist.Normal()

    vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist, x_dist = x_dist,
        include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss, mnist = mnist, d3 = d3, mpi = mpi, batch_size=args.batch_size)

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    print(num_iterations)
    print(args.num_epochs)
    print(len(train_loader))
    print(dataset_size)
    print("TYPE", type(train_loader.dataset[1]))

    trainer = Trainer(vae = vae, optimizer = optimizer,dataset_size = dataset_size, args = args, mnist = mnist)

    trainer.fit(train_loader = train_loader, num_iterations = num_iterations)
    
    return vae


if __name__ == '__main__':
    model = main()
