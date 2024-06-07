import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import utils.dist as dist
from utils.utils import logsumexp
from .d3_shapes_enc_dec import d3ConvEncoder, d3ConvDecoder
from .mnist_enc_dec import MnistEncoder, MnistDecoder
from .mlp_enc_dec import MLPEncoder, MLPDecoder
from .conv_enc_dec import ConvEncoder, ConvDecoder


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

        if isinstance(self.x_dist, dist.Normal) == True:
            if self.mnist == True:
                self.log_scale = nn.Parameter(torch.zeros(batch_size, 1, 28, 28,1))
                #self.log_scale = nn.Parameter(torch.zeros(batch_size, 1, 28, 28))
                #self.log_scale = torch.zeros(batch_size, 1, 28, 28,1).to(device='cuda')
            elif self.d3 == True or self.mpi == True:
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
