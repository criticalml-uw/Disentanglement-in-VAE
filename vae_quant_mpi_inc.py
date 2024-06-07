import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow
import torch.nn.init as init

from elbo_decomposition import elbo_decomposition
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)




class d3ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(d3ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),
            nn.ReLU(True),
            nn.Conv2d(256, self.output_dim, 1)
        )
        self.weight_init()
    
    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, x):
        h = x.view(-1, 3, 64, 64)
        z = self.encode(h)
        return z


class d3ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(d3ConvDecoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(input_dim, 256, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )
        self.weight_init()
    
    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        mu_img = self.decode(h)
        return mu_img

class MnistEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MnistEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc3 = nn.Linear(400, output_dim)
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 28 * 28)
        h = self.act(self.fc1(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z

class MnistDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MnistDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, 784)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 28, 28)
        return mu_img

class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(), x_dist = dist.Bernoulli(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False, mnist = False, d3 = False, mpi = False, batch_size = 32):
        super(VAE, self).__init__()

        self.mnist = mnist
        self.d3 = d3
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.mpi = mpi 
        #self.x_dist = dist.Bernoulli()

        self.x_dist = x_dist
        print("HEREEEEEEEEEEEEEEEEEEEEEEE", isinstance(self.x_dist, dist.Normal))
        if isinstance(self.x_dist, dist.Normal) == True:
            print("HEREEEEEEEEEEEEEEEEEEEEEEE")
            if self.mnist == True:
                self.log_scale = nn.Parameter(torch.zeros(batch_size, 1, 28, 28,1))
                #self.log_scale = nn.Parameter(torch.zeros(batch_size, 1, 28, 28))
                #self.log_scale = torch.zeros(batch_size, 1, 28, 28,1).to(device='cuda')
            elif self.d3 == True or self.mpi == True:
                print("**************************************", self.mpi)
                self.log_scale = nn.Parameter(torch.zeros(batch_size, 3, 64, 64,1))
                #self.log_scale = nn.Parameter(torch.zeros(batch_size, 3, 64, 64))
                #self.log_scale = torch.zeros(batch_size, 3, 64, 64,1).to(device='cuda')
            else:
                self.log_scale = nn.Parameter(torch.zeros(batch_size, 1, 64, 64,1))
                #self.log_scale = nn.Parameter(torch.zeros(batch_size, 1, 64, 64))
                #self.log_scale = torch.zeros(batch_size, 1, 64, 64,1).to(device='cuda')

            #self.fc_mu = nn.Linear(z_dim, z_dim)
            #self.fc_var = nn.Linear(z_dim, z_dim)

            self.mse_loss = nn.MSELoss(reduction = 'none')

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        elif self.mnist:
            self.encoder = MnistEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MnistDecoder(z_dim)
        elif self.d3:
            self.encoder = d3ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = d3ConvDecoder(z_dim)
        elif self.mpi:
            self.encoder = d3ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = d3ConvDecoder(z_dim)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        print("HERE", zs.shape)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        '''print("X SIZE",x.size())'''
        if self.mnist == True:
            x = x.view(x.size(0), 1, 28, 28)
        elif self.d3 == True or self.mpi == True:
            x = x.view(x.size(0), 3, 64, 64)
        else:
            x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        '''print("ZS", zs.shape)'''
        return zs, z_params

    def decode(self, z):
        if self.mnist == True:
            x_params = self.decoder.forward(z).view(z.size(0), 1, 28, 28)
        elif self.d3 == True or self.mpi == True:
            x_params = self.decoder.forward(z).view(z.size(0), 3, 64, 64)
        else:
            x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        
        '''print("XS", xs.shape)'''

        if isinstance(self.x_dist, dist.Normal) == True:
            x_params = torch.unsqueeze(x_params,4)
            x_params = torch.cat((x_params,self.log_scale), 4)

        xs = self.x_dist.sample(params=x_params)

        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        print("X_PARAMS", x_hat.shape)
        print("LOGSCALE", logscale.shape)
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        print("MU_STD", mu.shape, std.shape)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z).sum(-1)
        log_pz = p.log_prob(z).sum(-1)

        '''# kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)'''
        return log_pz, log_qzx
    
    def elbo_imp(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        if self.mnist == True:
            x = x.view(batch_size, 1, 28, 28)
        elif self.d3 == True:
            x = x.view(batch_size, 3, 64, 64)
        else:
            x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        print("Z PARAMS", z_params.shape)
        mu = z_params[:,:,0]
        std = z_params[:,:,1]
        mu = torch.exp(mu)
        std = torch.exp(std)
        logpx = self.gaussian_likelihood(x_params, self.log_scale, x)
        print("px", logpx[0])
        logpz, logqz_condx = self.kl_divergence(zs, mu, std)
        print("pz", logpz[0])
        print("qzx", logqz_condx[0])
        #print("x_dist_sigma", self.log_scale)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        if self.mnist == True:
            x = x.view(batch_size, 1, 28, 28)
        elif self.d3 == True or self.mpi == True:
            x = x.view(batch_size, 3, 64, 64)
        else:
            x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        '''if isinstance(self.x_dist, dist.Normal) == True:
            x_params = torch.unsqueeze(x_params,4)
            x_params = torch.cat((x_params,self.log_scale), 4)'''
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        #print("X X_RECONS", x.shape, x_recon.shape, x_params.shape)
        '''logpx = -self.mse_loss(x_recon,x).view(batch_size, -1).sum(1)'''
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        #print("px", logpx[0], "pz", logpz[0], "qzx", logqz_condx[0])
        #print("x_dist_sigma", self.log_scale)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False, test = False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    elif args.dataset == 'mnist' and test == False:
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        print(type(train_set))
    elif args.dataset == 'mnist' and test == True:
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.dataset == '3dshapes':
        train_set = dset.d3Shapes(test)
    elif args.dataset == 'mpi3D':
        train_set = dset.mpi3D(test)
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    '''kwargs = {'num_workers': 4, 'pin_memory': use_cuda}**kwargs'''
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True)
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis, mnist = False, d3 = False, mpi = False):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    if mnist == True:
        images = list(sample_mu.view(-1, 1, 28, 28).data.cpu())
    elif d3 == True or mpi == True:
        images = list(sample_mu.view(-1, 3, 64, 64).data.cpu())
    else:
        images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    if mnist == True:
        test_reco_imgs = torch.cat([
            test_imgs.view(1, -1, 28, 28), reco_imgs.view(1, -1, 28, 28)], 0).transpose(0, 1)
    elif d3 == True or mpi == True:
        test_reco_imgs = torch.cat([
            test_imgs.view(3, -1, 64, 64), reco_imgs.view(3, -1, 64, 64)], 0).transpose(0, 1)
    else:
        test_reco_imgs = torch.cat([
            test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)

    if mnist == True:
        win_test_reco = vis.images(
            list(test_reco_imgs.contiguous().view(-1, 1, 28, 28).data.cpu()), 10, 2,
            opts={'caption': 'test reconstruction image'}, win=win_test_reco)
    elif d3 == True or mpi == True:
        win_test_reco = vis.images(
            list(test_reco_imgs.contiguous().view(-1, 3, 64, 64).data.cpu()), 10, 2,
            opts={'caption': 'test reconstruction image'}, win=win_test_reco)
    else:
        win_test_reco = vis.images(
            list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()), 10, 2,
            opts={'caption': 'test reconstruction image'}, win=win_test_reco)

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


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
    print("MPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", mpi)
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

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=4500)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    print(num_iterations)
    print(args.num_epochs)
    print(len(train_loader))
    print(dataset_size)
    print("TYPE", type(train_loader.dataset[1]))
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            '''print("ITERATION", iteration)'''
            '''print("X SHAPE", type(x))'''
            '''print(x.size())'''
            if(mnist == True):
                '''print(type(x))
                print(type(x[0]))
                print(len(x))
                print(x[0].size())'''
                x = x[0]
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            x = x.cuda(non_blocking=True)
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            '''print("ELBO",elbo.mean())
            print("ELBO.data",elbo.data)
            print("ELBO.data[0]",elbo.data[0])
            print("OBJ", obj)
            print("OBJ DATA[0]",obj.data[0])
            print("OBJ.mean()",obj.mean())
            print("OBJ.mean().data[0]",obj.mean().data)'''
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().data)
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg))

                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis, mnist = mnist)
                    plot_elbo(train_elbo, vis)
                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, iteration)
                '''eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
                    os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))'''

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, iteration)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    '''eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))'''
    return vae


if __name__ == '__main__':
    model = main()
