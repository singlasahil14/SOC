import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import time
from einops import rearrange

from utils_conv import *

class skew_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, 
                 bias=True, train_terms=5, eval_terms=10, init_iters=50, update_iters=1, 
                 update_freq=200, correction=0.7):
        super(skew_conv, self).__init__()
        assert (stride==1) or (stride==2)
        self.init_iters = init_iters
        self.out_channels = out_channels
        self.in_channels = in_channels*stride*stride
        self.max_channels = max(self.out_channels, self.in_channels)
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.update_iters = update_iters
        self.update_freq = update_freq
        self.total_iters = 0
        self.train_terms = train_terms
        self.eval_terms = eval_terms
        
        if kernel_size == 1:
            correction = 1.0
        
        self.random_conv_filter = nn.Parameter(torch.Tensor(torch.randn(self.max_channels, 
                                               self.max_channels, self.kernel_size, 
                                               self.kernel_size)).cuda(),
                                               requires_grad=True)
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter = 0.5 * (self.random_conv_filter - random_conv_filter_T)
        
        with torch.no_grad():
            u1, v1, u2, v2, u3, v3, u4, v4 = fantastic_four(conv_filter, 
                                                num_iters=self.init_iters, 
                                                return_vectors=True)
            self.u1 = nn.Parameter(u1, requires_grad=False)
            self.v1 = nn.Parameter(v1, requires_grad=False)
            self.u2 = nn.Parameter(u2, requires_grad=False)
            self.v2 = nn.Parameter(v2, requires_grad=False)
            self.u3 = nn.Parameter(u3, requires_grad=False)
            self.v3 = nn.Parameter(v3, requires_grad=False)
            self.u4 = nn.Parameter(u4, requires_grad=False)
            self.v4 = nn.Parameter(v4, requires_grad=False)
            
        self.correction = nn.Parameter(torch.Tensor([correction]).cuda(), requires_grad=False)
            
        self.enable_bias = bias
        if self.enable_bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels).cuda(), requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()
            
    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.max_channels)
        nn.init.normal_(self.random_conv_filter, std=stdv)
        
        stdv = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)
            
    def update_sigma(self):
        if self.training:
            if self.total_iters % self.update_freq == 0:
                update_iters = self.init_iters
            else:
                update_iters = self.update_iters
            self.total_iters = self.total_iters + 1
        else:
            update_iters = 0
        
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter = 0.5*(self.random_conv_filter - random_conv_filter_T)
        pad_size = conv_filter.shape[2]//2
        with torch.no_grad():
            for i in range(update_iters):
                self.v1.data = l2_normalize((conv_filter*self.u1).sum(
                                            (1, 3), keepdim=True).data)
                self.u1.data = l2_normalize((conv_filter*self.v1).sum(
                                            (0, 2), keepdim=True).data)
                self.v2.data = l2_normalize((conv_filter*self.u2).sum(
                                            (1, 2), keepdim=True).data)
                self.u2.data = l2_normalize((conv_filter*self.v2).sum(
                                            (0, 3), keepdim=True).data)
                self.v3.data = l2_normalize((conv_filter*self.u3).sum(
                                            (1, 2, 3), keepdim=True).data)
                self.u3.data = l2_normalize((conv_filter*self.v3).sum(
                                            0, keepdim=True).data)
                self.v4.data = l2_normalize((conv_filter*self.u4).sum(
                                            (0, 2, 3), keepdim=True).data)
                self.u4.data = l2_normalize((conv_filter*self.v4).sum(
                                            1, keepdim=True).data)

        func = torch.min
        sigma1 = torch.sum(conv_filter*self.u1*self.v1)
        sigma2 = torch.sum(conv_filter*self.u2*self.v2)
        sigma3 = torch.sum(conv_filter*self.u3*self.v3)
        sigma4 = torch.sum(conv_filter*self.u4*self.v4)
        sigma = func(func(func(sigma1, sigma2), sigma3), sigma4)
        return sigma

    def forward(self, x):
        random_conv_filter_T = transpose_filter(self.random_conv_filter)
        conv_filter_skew = 0.5*(self.random_conv_filter - random_conv_filter_T)
        sigma = self.update_sigma()
        conv_filter_n = (self.correction * conv_filter_skew)/sigma
        
        if self.training:
            num_terms = self.train_terms
        else:
            num_terms = self.eval_terms
        
        if self.stride > 1:
            x = rearrange(x, "b c (w k1) (h k2) -> b (c k1 k2) w h", 
                          k1=self.stride, k2=self.stride)        
        
        if self.out_channels > self.in_channels:
            diff_channels = self.out_channels - self.in_channels
            p4d = (0, 0, 0, 0, 0, diff_channels, 0, 0)
            curr_z = F.pad(x, p4d)
        else:
            curr_z = x

        z = curr_z
        curr_fact = 1.
        for i in range(1, num_terms+1):
            curr_z = F.conv2d(curr_z, conv_filter_n, 
                              padding=(self.kernel_size//2, 
                                       self.kernel_size//2))/float(i)
            z = z + curr_z
            
        if self.out_channels < self.in_channels:
            z = z[:, :self.out_channels, :, :]
            
        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z
