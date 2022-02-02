import argparse
import copy
import logging
import os
import time
import math
from shutil import copyfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from lip_convnets import LipConvNet
from utils import *

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Training specifications
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--gamma', default=0., type=float, help='gamma for certificate regularization')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
        help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')

    
    # Model architecture specifications
    parser.add_argument('--conv-layer', default='soc', type=str, choices=['bcop', 'cayley', 'soc'], 
                        help='BCOP, Cayley, SOC convolution')
    parser.add_argument('--init-channels', default=32, type=int)
    parser.add_argument('--activation', default='maxmin', choices=['maxmin', 'hh1', 'hh2'], 
                        help='Activation function')
    parser.add_argument('--block-size', default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                        help='size of each block')
    parser.add_argument('--lln', action='store_true', help='set last linear to be linear and normalized')

    
    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset to use for training')
    
    # Other specifications
    parser.add_argument('--epsilon', default=36, type=int)
    parser.add_argument('--out-dir', default='LipConvnet', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def init_model(args):
    model = LipConvNet(args.conv_layer, args.activation, init_channels=args.init_channels, 
                       block_size = args.block_size, num_classes=args.num_classes, 
                       lln=args.lln)
    return model

def robust_statistics(losses_arr, correct_arr, certificates_arr, 
                      epsilon_list=[36., 72., 108., 144., 180., 216.]):
    mean_loss = np.mean(losses_arr)
    mean_acc = np.mean(correct_arr)
    mean_certs = (certificates_arr * correct_arr).sum()/correct_arr.sum()
    
    robust_acc_list = []
    for epsilon in epsilon_list:
        robust_correct_arr = (certificates_arr > (epsilon/255.)) & correct_arr
        robust_acc = robust_correct_arr.sum()/robust_correct_arr.shape[0]
        robust_acc_list.append(robust_acc)
    return mean_loss, mean_acc, mean_certs, robust_acc_list

def main():
    args = get_args()
    
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    args.out_dir += '_' + str(args.dataset) 
    args.out_dir += '_' + str(args.block_size) 
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.init_channels)
    args.out_dir += '_' + str(args.activation)
    args.out_dir += '_cr' + str(args.gamma)
    if args.lln:
        args.out_dir += '_lln'
    
    
    os.makedirs(args.out_dir, exist_ok=True)
    code_dir = os.path.join(args.out_dir, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for f in os.listdir('./'):
        src = os.path.join('./', f)
        dst = os.path.join(code_dir, f)
        if os.path.isfile(src):
            if f[-3:] == '.py' or f[-3:] == '.sh':
                copyfile(src, dst)
    
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset)
    std = cifar10_std
    if args.dataset == 'cifar10':
        args.num_classes = 10    
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        raise Exception('Unknown dataset')

    # Evaluation at early stopping
    model = init_model(args).cuda()
    model.train()

    conv_params, activation_params, other_params = parameter_lists(model)
    if args.conv_layer == 'soc':
        opt = torch.optim.SGD([
                        {'params': activation_params, 'weight_decay': 0.},
                        {'params': (conv_params + other_params), 'weight_decay': args.weight_decay}
                    ], lr=args.lr_max, momentum=args.momentum)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, 
                              weight_decay=0.)
        
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = True
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, 
        (3 * lr_steps) // 4], gamma=0.1)
    
    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')
    
    # Training
    std = torch.tensor(std).cuda()
    L = 1/torch.max(std)
    prev_robust_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t ' + 
                'Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Test Cert')
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_cert = 0
        train_robust = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            output = model(X)
            curr_correct = (output.max(1)[1] == y)
            if args.lln:
                curr_cert = lln_certificates(output, y, model.last_layer, L)
            else:
                curr_cert = ortho_certificates(output, y, L)
                
            ce_loss = criterion(output, y)
            loss = ce_loss - args.gamma * F.relu(curr_cert).mean()

            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()

            train_loss += ce_loss.item() * y.size(0)
            train_cert += (curr_cert * curr_correct).sum().item()
            train_robust += ((curr_cert > (args.epsilon/255.)) * curr_correct).sum().item()
            train_acc += curr_correct.sum().item()
            train_n += y.size(0)
            scheduler.step()
            
        # Check current test accuracy of model
        losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model, L)
        
        test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
            losses_arr, correct_arr, certificates_arr)
        
        robust_acc = test_robust_acc_list[0]
        if (robust_acc >= prev_robust_acc):
            torch.save(model.state_dict(), best_model_path)
            prev_robust_acc = robust_acc
            best_epoch = epoch
        
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, 
            test_loss, test_acc, test_robust_acc_list[0], test_robust_acc_list[1], 
            test_robust_acc_list[2], test_cert)
        
        torch.save(model.state_dict(), last_model_path)
        
        trainer_state_dict = { 'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)
        
    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    
    # Evaluation at best model (early stopping)
    model_test = init_model(args).cuda()
    model_test.load_state_dict(torch.load(best_model_path))
    model_test.float()
    model_test.eval()
        
    start_test_time = time.time()
    losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
        losses_arr, correct_arr, certificates_arr)
    
    logger.info('Best Epoch \t Test Loss \t Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Mean Cert \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', best_epoch, test_loss, test_acc,
                                                        test_robust_acc_list[0], test_robust_acc_list[1], 
                                                        test_robust_acc_list[2], test_cert, total_time)

    # Evaluation at last model
    model_test.load_state_dict(torch.load(last_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    losses_arr, correct_arr, certificates_arr = evaluate_certificates(test_loader, model_test, L)
    total_time = time.time() - start_test_time
    
    test_loss, test_acc, test_cert, test_robust_acc_list = robust_statistics(
        losses_arr, correct_arr, certificates_arr)
    
    logger.info('Last Epoch \t Test Loss \t Test Acc \t Test Robust (36) \t Test Robust (72) \t Test Robust (108) \t Mean Cert \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', epoch, test_loss, test_acc,
                                                        test_robust_acc_list[0], test_robust_acc_list[1], 
                                                        test_robust_acc_list[2], test_cert, total_time)

if __name__ == "__main__":
    main()


