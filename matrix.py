import torch
import torch.nn as nn
from utils.initialization import kaiming_init, normal_init

class Matrix(nn.Module):
    def __init__(self, input_dimension, output_dimension,intermediate_dimension, non_linearity = False, use_cuda = False, channel_dimension = 1, initial = False):
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
        if initial:
            self.weight_init()

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
            return ep_img, ep_img

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for m in self.modules():
            print("PRINTING MODULE INSIDE MODEL", m)
            initializer(m)
