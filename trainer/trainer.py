from utils.utils import plot_elbo
import utils.utils as utils
import os
import time
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.display_image_win import display_samples
import visdom
from utils.anneal_kl import anneal_kl
from elbo_decomposition import elbo_decomposition


class Trainer:
    def __init__(self, vae, optimizer,dataset_size, args, mnist):
        self.vae = vae
        self.optimizer = optimizer
        self.dataset_size = dataset_size
        self.args = args
        self.mnist = mnist



    def train_one_iteration(self, elbo_running_mean, iteration):
        if(self.mnist == True):
                x = x[0]
        iteration += 1
        batch_time = time.time()
        self.vae.train()
        anneal_kl(self.args, self.vae, iteration)
        self.optimizer.zero_grad()
        # transfer to GPU
        x = x.cuda(non_blocking=True)
        # wrap the mini-batch in a PyTorch Variable
        x = Variable(x)
        # do ELBO gradient and accumulate loss
        obj, elbo = self.vae.elbo(x, self.dataset_size)
        if utils.isnan(obj).any():
            raise ValueError('NaN spotted in objective.')
        obj.mean().mul(-1).backward()
        elbo_running_mean.update(elbo.mean().data)
        self.optimizer.step()

        return elbo_running_mean, batch_time
    
    def statistics_after_training(self, iteration, train_loader):
        self.vae.eval()
        utils.save_checkpoint({
            'state_dict': self.vae.state_dict(),
            'args': self.args}, self.args.save, iteration)
        dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, shuffle=False)
        logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
            elbo_decomposition(self.vae, dataset_loader)
        torch.save({
            'logpx': logpx,
            'dependence': dependence,
            'information': information,
            'dimwise_kl': dimwise_kl,
            'analytical_cond_kl': analytical_cond_kl,
            'marginal_entropies': marginal_entropies,
            'joint_entropy': joint_entropy
        }, os.path.join(self.args.save, 'elbo_decomposition.pth'))
        '''eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'))'''
    
    def fit(self, train_loader, num_iterations):

        if self.args.visdom:
            vis = visdom.Visdom(env=self.args.save, port=4500)
    
        train_elbo = []
        elbo_running_mean = utils.RunningAverageMeter()
        iteration = 0
        while iteration < num_iterations:
            for i, x in enumerate(train_loader):
                elbo_running_mean, batch_time = self.train_one_iteration(elbo_running_mean, iteration)

                if iteration % self.args.log_freq == 0:
                    train_elbo.append(elbo_running_mean.avg)
                    print('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)' % (
                        iteration, time.time() - batch_time, self.vae.beta, self.vae.lamb,
                        elbo_running_mean.val, elbo_running_mean.avg))
                    

                    self.vae.eval()

                    # plot training and test ELBOs
                    if self.args.visdom:
                        display_samples(self.vae, x, vis, mnist = self.mnist)
                        plot_elbo(train_elbo, vis)
                    utils.save_checkpoint({
                        'state_dict': self.vae.state_dict(),
                        'args': self.args}, self.args.save, iteration)

                
    