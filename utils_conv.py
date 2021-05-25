import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
import time

from matplotlib import pyplot as plt

def conv_singular_values_numpy(kernel, input_shape):
    _, _, ker_h, ker_w = kernel.shape
    kernel = np.transpose(kernel, [2, 3, 0, 1])
    transforms = np.fft.fft2(kernel, (ker_h, ker_w), axes=[0, 1])
    conv_svs = np.linalg.svd(transforms, compute_uv=False)
    print('real conv svs stats: ', conv_svs.min(), conv_svs.mean(), conv_svs.max())
    return conv_svs

def grads_cat(z_flat, x_flat):
    z_size = z_flat.shape[1]
    grads = []
    for i in range(z_size):
        x_grad = torch.autograd.grad(z_flat[0, i], x_flat, retain_graph=True)[0]
        grads.append(x_grad)
    return torch.cat(grads, dim=0)

def jacobian_conv(x, conv_filter, conv_type='circulant'):
    assert conv_type in ['circulant', 'toeplitz']
    batch_size = x.shape[0]
    assert batch_size == 1
    _, _, h, w = conv_filter.shape
    
    x_flat = x.view(batch_size, -1)
    x_ = x_flat.view_as(x)
    
    if conv_type == 'circulant':
        p4d = (h//2, h//2, w//2, w//2)
        x_pad = F.pad(x_, p4d, mode='circular')
        z = F.conv2d(x_pad, conv_filter)
    elif conv_type == 'toeplitz':
        z = F.conv2d(x_, conv_filter, padding=(h//2, w//2))
    else:
        raise ValueError('Unknown convolution type: {%s}'.format(conv_type))
        
    z_flat = z.view(batch_size, -1)
    return grads_cat(z_flat, x_flat)
    
def check_skew_symmetric(A):
    neg_A_T = -A.t()
    print(torch.max(A - neg_A_T))
    
def check_orthogonality_series(A, num_terms=10):
    m, n = A.shape
    assert m==n
    I = torch.eye(n).cuda()
    exp_A = I
    curr_A = A.clone()
    fact = 1
    for i in range(1, num_terms):
        fact = fact*i
        exp_A = exp_A + (curr_A/fact)
        curr_A = curr_A@A
        print(torch.max(exp_A@exp_A.t() - I))

def check_identity(A):
    identity = torch.eye(A.shape[0]).cuda()
    diff_abs = torch.abs(A - identity)
    sum_val = torch.sum(diff_abs*diff_abs).item()
    max_val = torch.max(diff_abs).item()
    print(sum_val, max_val)
    
def check_orthogonality(A, suffix=''):
    m, n = A.shape[-2:]
    if m > n:
        identity = (A.transpose(-1, -2)@A).cuda()
        identity_custom = torch.eye(n).cuda()
        diff_abs = torch.abs(identity - identity_custom)
        sum_val = torch.sum(diff_abs*diff_abs).item()
        max_val = torch.max(diff_abs).item()
        print(suffix, sum_val, max_val)
        
    if m <= n:
        identity = (A@A.transpose(-1, -2)).cuda()
        identity_custom = torch.eye(m).cuda()
        diff_abs = torch.abs(identity - identity_custom)
        sum_val = torch.sum(diff_abs*diff_abs).item()
        max_val = torch.max(diff_abs).item()
        print(suffix, sum_val, max_val)

def transpose_filter(conv_filter):
    conv_filter_T = torch.transpose(conv_filter, 0, 1)    
    conv_filter_T = torch.flip(conv_filter_T, [2, 3])
    return conv_filter_T

def random_filter(num_ch, h, w, requires_grad=True):
    conv_filter = torch.randn(num_ch, num_ch, h, w, requires_grad=requires_grad).cuda()
    return conv_filter

def random_input(num_ch, H, W, requires_grad=True, flatten=False):
    random_inp = torch.randn(1, num_ch, H, W, requires_grad=requires_grad).cuda()
    if flatten:
        random_inp = random_inp.view(1, -1)
    return random_inp

def skew_symmetric_filter(num_ch, h, w, requires_grad=True):
    conv_filter = random_filter(num_ch, h, w, requires_grad=requires_grad)
    conv_filter_T = transpose_filter(conv_filter)

    conv_filter_skew = 0.5*(conv_filter - conv_filter_T)
    return conv_filter_skew

def identity_filter(num_ch, h, w, requires_grad=True):
    conv_filter = torch.zeros(num_ch, num_ch, h, w, requires_grad=requires_grad).cuda()
    mid_h, mid_w = h//2, w//2
    for i in range(num_ch):
        conv_filter[i, i, mid_h, mid_w] = 1.
    return conv_filter

def test_orthogonality_lip_skew(model):
    model_l = [module for module in model.modules() if type(module) != nn.Sequential]
    for module in model_l:
        if str(type(module))=="<class 'skew_symmetric_conv.skew_conv'>":
            print('convolution layer')
            module.num_terms = 15
            in_ch = module.in_channels // (module.stride * module.stride)
            out_ch = module.out_channels

            H, W = 16, 16
            x_flat = random_input(in_ch, H, W, requires_grad=True, flatten=True)
            x = x_flat.view(1, in_ch, H, W)
            
            start_time = time.time()
            z = module(x)
            print(time.time() - start_time, x.shape, z.shape)

            z_flat = z.view(1, -1)
            J = grads_cat(z_flat, x_flat)
            
            m, n = J.shape
            if n>=m:
                mat = J@J.t()
                check_identity(mat)
            if m>=n:
                mat = J.t()@J
                check_identity(mat)
        elif str(type(module))=="<class 'bjorck_linear.BjorckLinear'>":
            print('linear layer')
            J = module.ortho_w()
            m, n = J.shape
            if n>=m:
                mat = J@J.t()
                check_identity(mat)
            if m>=n:
                mat = J.t()@J
                check_identity(mat)
        else:
            pass
        
def test_orthogonality_lip_bcop(model):
    model_l = [module for module in model.modules() if type(module) != nn.Sequential]
    for module in model_l:
        if str(type(module))=="<class 'lip_block_orthogonal_conv.bcop_conv'>":
            print('convolution layer')
            in_ch = module.in_channels // module.stride // module.stride
            out_ch = module.out_channels

            H, W = 16, 16
            x_flat = random_input(in_ch, H, W, requires_grad=True, flatten=True)
            x = x_flat.view(1, in_ch, H, W)
            
            start_time = time.time()
            z = module(x)
            print(time.time() - start_time, x.shape, z.shape)

            z_flat = z.view(1, -1)
            J = grads_cat(z_flat, x_flat)
            
            m, n = J.shape
            if n>=m:
                mat = J@J.t()
                check_identity(mat)
            if m>=n:
                mat = J.t()@J
                check_identity(mat)
        elif str(type(module))=="<class 'bjorck_linear.BjorckLinear'>":
            print('linear layer')
            J = module.ortho_w()
            m, n = J.shape
            if n>=m:
                mat = J@J.t()
                check_identity(mat)
            if m>=n:
                mat = J.t()@J
                check_identity(mat)
        else:
            pass

def test_orthogonality_skew(model):
    model_l = [module for module in model.modules() if type(module) != nn.Sequential]
    for module in model_l:
        if str(type(module))=="<class 'skew_symmetric_conv.skew_conv'>":
            in_ch = module.in_channels
            out_ch = module.out_channels

            H, W = 16, 16
            x_flat = random_input(in_ch, H, W, requires_grad=True, flatten=True)
            x = x_flat.view(1, in_ch, H, W)
                        
            start_time = time.time()
            z = module(x)
            print(time.time() - start_time, x.shape, z.shape)

            z_flat = z.view(1, -1)
            J = grads_cat(z_flat, x_flat)
            m, n = J.shape
            print(m, n)
            if n>=m:
                mat = J@J.t()
                print(mat.shape)
                check_identity(mat)
            if m>=n:
                mat = J.t()@J
                print(mat.shape)
                check_identity(mat)
                
def test_orthogonality_bcop(model):
    pass
#     model_l = [module for module in model.modules() if type(module) != nn.Sequential]
#     for module in model_l:
#         if str(type(module))=="<class 'skew_symmetric_conv.skew_conv'>":
#             in_ch = module.in_channels
#             out_ch = module.out_channels

#             H, W = 16, 16
#             x_flat = random_input(in_ch, H, W, requires_grad=True, flatten=True)
#             x = x_flat.view(1, in_ch, H, W)
            
#             module.num_terms = 20
#             conv_filter = module.conv_filter
#             sigma = module.update_sigma()
#             conv_filter_n = ((module.correction * conv_filter)/sigma)
#             sigma_n = real_power_iteration(conv_filter_n, num_iters=50)
#             print(sigma_n.detach().cpu().item())

def fantastic_four(conv_filter, num_iters=50, return_vectors=False):
    out_ch, in_ch, h, w = conv_filter.shape
    
    u1 = torch.randn((1, in_ch, 1, w), device='cuda', requires_grad=False)
    u1.data = l2_normalize(u1.data)

    u2 = torch.randn((1, in_ch, h, 1), device='cuda', requires_grad=False)
    u2.data = l2_normalize(u2.data)

    u3 = torch.randn((1, in_ch, h, w), device='cuda', requires_grad=False)
    u3.data = l2_normalize(u3.data)

    u4 = torch.randn((out_ch, 1, h, w), device='cuda', requires_grad=False)
    u4.data = l2_normalize(u4.data)
        
    v1 = torch.randn((out_ch, 1, h, 1), device='cuda', requires_grad=False)
    v1.data = l2_normalize(v1.data)

    v2 = torch.randn((out_ch, 1, 1, w), device='cuda', requires_grad=False)
    v2.data = l2_normalize(v2.data)

    v3 = torch.randn((out_ch, 1, 1, 1), device='cuda', requires_grad=False)
    v3.data = l2_normalize(v3.data)

    v4 = torch.randn((1, in_ch, 1, 1), device='cuda', requires_grad=False)
    v4.data = l2_normalize(v4.data)

    for i in range(num_iters):
        v1.data = l2_normalize((conv_filter.data*u1.data).sum((1, 3), keepdim=True).data)
        u1.data = l2_normalize((conv_filter.data*v1.data).sum((0, 2), keepdim=True).data)
        
        v2.data = l2_normalize((conv_filter.data*u2.data).sum((1, 2), keepdim=True).data)
        u2.data = l2_normalize((conv_filter.data*v2.data).sum((0, 3), keepdim=True).data)
        
        v3.data = l2_normalize((conv_filter.data*u3.data).sum((1, 2, 3), keepdim=True).data)
        u3.data = l2_normalize((conv_filter.data*v3.data).sum(0, keepdim=True).data)
        
        v4.data = l2_normalize((conv_filter.data*u4.data).sum((0, 2, 3), keepdim=True).data)
        u4.data = l2_normalize((conv_filter.data*v4.data).sum(1, keepdim=True).data)

    sigma1 = torch.sum(conv_filter.data*u1.data*v1.data)
    sigma2 = torch.sum(conv_filter.data*u2.data*v2.data)
    sigma3 = torch.sum(conv_filter.data*u3.data*v3.data)
    sigma4 = torch.sum(conv_filter.data*u4.data*v4.data)

    if return_vectors:
        return u1, v1, u2, v2, u3, v3, u4, v4
    else:
        return sigma1, sigma2, sigma3, sigma4
    
def l2_normalize(tensor, eps=1e-12):
    norm = float(torch.sqrt(torch.sum(tensor.float() * tensor.float())))
    norm = max(norm, eps)
    ans = tensor / norm
    return ans

def real_power_iteration(conv_filter, inp_shape=(32, 32), num_iters=50, return_uv=False):
    H, W = inp_shape
    c_out = conv_filter.shape[0]
    c_in = conv_filter.shape[1]
    pad_size = conv_filter.shape[2]//2
    with torch.no_grad():
        u = l2_normalize(torch.randn(1, c_out, H, W, dtype=conv_filter.dtype).cuda().data)
        v = l2_normalize(torch.randn(1, c_in, H, W, dtype=conv_filter.dtype).cuda().data)

        for i in range(num_iters):
            v.data = l2_normalize(F.conv_transpose2d(u.data, conv_filter.data, padding=pad_size))
            u.data = l2_normalize(F.conv2d(v.data, conv_filter.data, padding=pad_size))
        sigma = torch.sum(u.data * F.conv2d(v.data, conv_filter.data, padding=pad_size))
    if return_uv:
        return sigma, u, v
    else:
        return sigma

def test_real_sn(model):
    model_l = [module for module in model.modules() if type(module) != nn.Sequential]
    sigma_list = []
    for module in model_l:
        if str(type(module)) in ["<class 'skew_symmetric_conv.skew_conv'>", "<class 'lip_skew_symmetric_conv.skew_conv'>"]:
            in_ch = module.in_channels
            out_ch = module.out_channels
            
            conv_filter = module.random_conv_filter
            conv_filter_T = transpose_filter(conv_filter)
            conv_filter_skew = 0.5*(conv_filter - conv_filter_T)

            real_sigma = module.update_sigma()
            conv_filter_n = ((module.correction * conv_filter_skew)/real_sigma)
            
            real_sigma = real_power_iteration(conv_filter_n, num_iters=50)
            
            sigma_list.append(real_sigma.detach().cpu().item())
    sigma_array = np.array(sigma_list)
    return sigma_array

def power_iteration(A, init_u=None, n_iters=20, return_uv=True):
    shape = list(A.shape)
    shape[-1] = 1
    shape = tuple(shape)
    if init_u is None:
        u = torch.randn(*shape, dtype=A.dtype, device=A.device)
    else:
        assert tuple(init_u.shape) == shape, (init_u.shape, shape)
        u = init_u
    for _ in range(n_iters):
        v = A.transpose(-1, -2) @ u
        v /= v.norm(dim=-2, keepdim=True)
        u = A @ v
        u /= u.norm(dim=-2, keepdim=True)
    s = (u.transpose(-1, -2) @ A @ v).squeeze(-1).squeeze(-1)
    if return_uv:
        return u, s, v
    return s

def bjorck_orthonormalize(
        w, beta=0.5, iters=20, order=1, power_iteration_scaling=False,
        default_scaling=False):
    if w.shape[-2] < w.shape[-1]:
        return bjorck_orthonormalize(
            w.transpose(-1, -2),
            beta=beta, iters=iters, order=order,
            power_iteration_scaling=power_iteration_scaling,
            default_scaling=default_scaling).transpose(
            -1, -2)

    if power_iteration_scaling:
        with torch.no_grad():
            s = power_iteration(w, return_uv=False)
        w = w / s.unsqueeze(-1).unsqueeze(-1)
    elif default_scaling:
        w = w / ((w.shape[0] * w.shape[1]) ** 0.5)
    assert order == 1, "only first order Bjorck is supported"
    for _ in range(iters):
        w_t_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ w_t_w
    return w

def orthogonal_matrix(n):
    a = torch.randn((n, n))
    q, r = torch.qr(a)
    return q * torch.sign(torch.diag(r))

def symmetric_projection(n, ortho_matrix, mask=None):
    q = ortho_matrix
    # randomly zeroing out some columns
    if mask is None:
        mask = (torch.randn(n) > 0).float()
    c = q * mask
    return c.mm(c.t())

def block_orth(p1, p2):
    assert p1.shape == p2.shape
    n = p1.size(0)
    kernel2x2 = {}
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    kernel2x2[0, 0] = p1.mm(p2)
    kernel2x2[0, 1] = p1.mm(eye - p2)
    kernel2x2[1, 0] = (eye - p1).mm(p2)
    kernel2x2[1, 1] = (eye - p1).mm(eye - p2)
    return kernel2x2

def matrix_conv(m1, m2):
    n = (m1[0, 0]).size(0)
    if n != (m2[0, 0]).size(0):
        raise ValueError(
            "The entries in matrices m1 and m2 " "must have the same dimensions!"
        )
    k = int(np.sqrt(len(m1)))
    l = int(np.sqrt(len(m2)))
    result = {}
    size = k + l - 1
    # Compute matrix convolution between m1 and m2.
    for i in range(size):
        for j in range(size):
            result[i, j] = torch.zeros(
                (n, n), device=m1[0, 0].device, dtype=m1[0, 0].dtype
            )
            for index1 in range(min(k, i + 1)):
                for index2 in range(min(k, j + 1)):
                    if (i - index1) < l and (j - index2) < l:
                        result[i, j] += m1[index1, index2].mm(
                            m2[i - index1, j - index2]
                        )
    return result

def dict_to_tensor(x, k1, k2):
    return torch.stack([torch.stack([x[i, j] for j in range(k2)]) for i in range(k1)])

def convolution_orthogonal_generator_projs(ksize, cin, cout, ortho, sym_projs):
    flipped = False
    if ksize == 1:
        return ortho.t().unsqueeze(-1).unsqueeze(-1)
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
        ortho = ortho.t()
    p = block_orth(sym_projs[0], sym_projs[1])
    for _ in range(1, ksize - 1):
        p = matrix_conv(p, block_orth(sym_projs[_ * 2], sym_projs[_ * 2 + 1]))
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = ortho.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)

def convolution_orthogonal_generator(ksize, cin, cout, P, Q):
    flipped = False
    if cin > cout:
        flipped = True
        cin, cout = cout, cin
    orth = orthogonal_matrix(cout)[0:cin, :]
    if ksize == 1:
        return orth.unsqueeze(0).unsqueeze(0)

    p = block_orth(symmetric_projection(cout, P[0]), symmetric_projection(cout, Q[0]))
    for _ in range(ksize - 2):
        temp = block_orth(
            symmetric_projection(cout, P[_ + 1]), symmetric_projection(cout, Q[_ + 1])
        )
        p = matrix_conv(p, temp)
    for i in range(ksize):
        for j in range(ksize):
            p[i, j] = orth.mm(p[i, j])
    if flipped:
        return dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0)
    return dict_to_tensor(p, ksize, ksize).permute(3, 2, 1, 0)

def convolution_orthogonal_initializer(ksize, cin, cout):
    P, Q = [], []
    cmax = max(cin, cout)
    for i in range(ksize - 1):
        P.append(orthogonal_matrix(cmax))
        Q.append(orthogonal_matrix(cmax))
    P, Q = map(torch.stack, (P, Q))
    return convolution_orthogonal_generator(ksize, cin, cout, P, Q)
