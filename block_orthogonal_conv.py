import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import time
from einops import rearrange

from utils_conv import *

class bcop_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, 
                 bias=True, bjorck_beta=0.5, bjorck_iters=30, bjorck_order=1, 
                 power_iteration_scaling=True):
        super(bcop_conv, self).__init__()
        assert (stride==1) or (stride==2)
        self.in_channels = in_channels*stride*stride
        self.out_channels = out_channels
        
        self.max_channels = max(self.out_channels, self.in_channels)
        
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_kernels = 2*self.kernel_size - 1

        self.bjorck_iters = bjorck_iters
        self.bjorck_beta = bjorck_beta
        self.bjorck_order = bjorck_order
        
        self.buffer_weight = None

        self.power_iteration_scaling = power_iteration_scaling
        
        # Define the unconstrained matrices Ms and Ns for Ps and Qs
        self.conv_matrices = nn.Parameter(
            torch.Tensor(self.num_kernels, self.max_channels, 
                         self.max_channels),
            requires_grad=True,
        )

        # The mask controls the rank of the symmetric projectors (full half rank).
        self.mask = nn.Parameter(
            torch.cat(
                (
                    torch.ones(self.num_kernels - 1, 1, self.max_channels // 2),
                    torch.zeros(
                        self.num_kernels - 1, 1,
                        self.max_channels - (self.max_channels // 2),
                    ),
                ),
                dim=-1,
            ),
            requires_grad=False,
        )

        # Bias parameters in the convolution
        self.enable_bias = bias
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(self.out_channels).cuda(), requires_grad=True
            )
        else:
            self.bias = None

        # Initialize the weights (self.weight is set to zero for streamline module)
        self.reset_parameters()

    def reset_parameters(self):
        for index in range(self.num_kernels):
            nn.init.orthogonal_(self.conv_matrices[index])

        stdv = 1.0 / np.sqrt(self.out_channels)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        self._input_shape = x.shape[2:]
        
        # orthogonalize all the matrices using Bjorck
        if self.training or self.buffer_weight is None:
            ortho = bjorck_orthonormalize(
                self.conv_matrices,
                beta=self.bjorck_beta,
                iters=self.bjorck_iters,
                power_iteration_scaling=self.power_iteration_scaling,
                default_scaling=not self.power_iteration_scaling,
            )

            # compute the symmetric projectors
            H = ortho[0, :self.in_channels, :self.out_channels]
            PQ = ortho[1:]
            if self.kernel_size > 1:
                PQ = PQ * self.mask
                PQ = PQ @ PQ.transpose(-1, -2)

            # compute the resulting convolution kernel using block convolutions
            weight = convolution_orthogonal_generator_projs(
                self.kernel_size, self.in_channels, self.out_channels, H, PQ
            )
            self.buffer_weight = weight
        else:
            weight = self.buffer_weight

        # detach the weight when we are using the cached weights from previous steps
        bias = self.bias
        
        # apply cyclic padding to the input and perform a standard convolution
        x_orig = x
        if self.stride > 1:
            x = rearrange(x, "b c (w k1) (h k2) -> b (c k1 k2) w h", 
                          k1=self.stride, k2=self.stride)

        p4d = (self.kernel_size//2, self.kernel_size//2, 
               self.kernel_size//2, self.kernel_size//2)
        x_pad = F.pad(x, p4d, mode='circular')

        z = F.conv2d(x_pad, weight)
        if self.enable_bias:
            z = z + self.bias.view(1, -1, 1, 1)
        return z

    def extra_repr(self):
        return "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, bias={enable_bias}, mask_half={mask_half}, ortho_mode={ortho_mode}".format(
            **self.__dict__
        )
