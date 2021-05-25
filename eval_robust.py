import argparse
import copy
import logging
import os
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from robust_net import LipNet_n
from utils import (upper_limit, lower_limit, cifar10_mean, cifar10_std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, evaluate_certificates)
from utils_conv import test_real_sn

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=0., type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=36, type=int)
    parser.add_argument('--in-channels', default=16, type=int)
    parser.add_argument('--block-size', default=1, type=int, help='model type')
    parser.add_argument('--out-dir', default='robust', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--conv-type', default='skew', type=str, choices=['standard', 'bcop', 
        'skew'], help='standard, skew symmetric or bcop convolution')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset to use for training')
    parser.add_argument('--beta', default=0.0, type=float, help='beta for regularization')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    return parser.parse_args()


def main():
    args = get_args()
    
    if args.dataset == 'cifar10':
        args.in_channels = 32
    elif args.dataset == 'cifar100':
        args.in_channels = 32
    else:
        raise Exception('Unknown dataset ', args.dataset)
    args.out_dir = args.out_dir + '_' + str(args.block_size) + '/lipnet_' + str(args.dataset) + '_' + str(args.block_size) + '_' + str(args.conv_type) + '_' + str(args.beta)
    print(args.out_dir)
    
#     os.makedirs(args.out_dir, exist_ok=True)
#     logfile = os.path.join(args.out_dir, 'output.log')
#     if os.path.exists(logfile):
#         os.remove(logfile)

#     logging.basicConfig(
#         format='%(message)s',
#         level=logging.INFO,
#         filename=os.path.join(args.out_dir, 'output.log'))
#     logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset)
    im_size = 32
    std = cifar10_std
    if args.dataset == 'cifar10':
        num_classes = 10    
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise Exception('Unknown dataset')
        
    std = torch.tensor(std).cuda()
    L = 1/torch.max(std)

    
    start_test_time = time.time()
    # Evaluation at early stopping
    model_test = LipNet_n(args.conv_type, in_channels=args.in_channels, num_blocks=args.block_size,
                          num_classes=num_classes).cuda()
    model_test.load_state_dict(torch.load(os.path.join(args.out_dir, 'best.pth')))
    model_test.float()
    model_test.eval()
        
    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model_test, L, args.epsilon)
    total_time = time.time() - start_test_time
    
    print('Test Loss \t Test Acc \t Robust Acc \t Mean Cert \t Test Time')
    print('{:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(test_loss, 
                                                      test_acc, robust_acc, 
                                                      mean_cert, total_time))

    # Evaluation at last model
    model_test.load_state_dict(torch.load(os.path.join(args.out_dir, 'last.pth')))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc, mean_cert, robust_acc = evaluate_certificates(test_loader, model_test, L, args.epsilon)
    total_time = time.time() - start_test_time
    
    print('Test Loss \t Test Acc \t Robust Acc \t Mean Cert \t Test Time')
    print('{:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(test_loss, 
                                                      test_acc, robust_acc, 
                                                      mean_cert, total_time))


if __name__ == "__main__":
    main()



