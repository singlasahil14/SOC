import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp

from preactresnet import *
from utils import (upper_limit, lower_limit, cifar10_mean, cifar10_std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Training specifications
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O2'],
        help='O0 is FP32 training and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')

    # Model architecture specifications
    parser.add_argument('--model-name', default='resnet18', type=str, choices=['resnet18', 'resnet34', 
        'resnet50', 'resnet101', 'resnet152'], help='Resnet model architecture to use')
    parser.add_argument('--conv-layer', default='standard', type=str, choices=['standard', 'bcop', 
        'cayley', 'soc'], help='Standard, BCOP, Cayley, SOC convolution')
    parser.add_argument('--activation', default='relu', choices=['relu', 'swish', 'maxmin', 
        'hh1', 'hh2'], help='Activation function')
    
    # Dataset specifications
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset to use for training')
    
    # Other specifications
    parser.add_argument('--out-dir', default='test/standard', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

def init_model(args):
    if args.dataset == 'cifar10':
        num_classes = 10    
    elif args.dataset == 'cifar100':
        num_classes = 100
    
    model_func = resnet_mapping[args.model_name]
    model = model_func(conv_name=args.conv_layer, activation_name=args.activation, 
                       num_classes=num_classes)
    return model

def main():
    args = get_args()
    
    if args.conv_layer == 'cayley' and args.opt_level == 'O2':
        raise ValueError('O2 optimization level is incompatible with Cayley Convolution')

    args.out_dir += '_' + str(args.dataset)
    args.out_dir += '_' + str(args.model_name)
    args.out_dir += '_' + str(args.conv_layer)
    args.out_dir += '_' + str(args.activation)
        
    os.makedirs(args.out_dir, exist_ok=True)
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

    model = init_model(args).cuda()
    model.train()

    
    conv_params = []
    activation_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'activation' in name:
                activation_params.append(param)
            elif 'conv' in name:
                conv_params.append(param)
            else:
                other_params.append(param)
    
    if args.conv_layer in ['standard', 'soc']:
        opt = torch.optim.SGD([
                        {'params': activation_params, 'weight_decay': 0.},
                        {'params': (conv_params + other_params), 'weight_decay': args.weight_decay}
                    ], lr=args.lr_max, momentum=args.momentum)
    else:
        opt = torch.optim.SGD([
                                {'params': (conv_params + activation_params), 'weight_decay': 0.},
                                {'params': other_params, 'weight_decay': args.weight_decay}
                              ], lr=args.lr_max, momentum=args.momentum)
        
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = True
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps // 2, 
            (3 * lr_steps) // 4], gamma=0.1)
        
    best_model_path = os.path.join(args.out_dir, 'best.pth')
    last_model_path = os.path.join(args.out_dir, 'last.pth')
    last_opt_path = os.path.join(args.out_dir, 'last_opt.pth')
        
    # Training
    prev_test_acc = 0.
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t Test Loss \t Test Acc')
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            
            output = model(X)
            ce_loss = criterion(output, y)

            opt.zero_grad()
            with amp.scale_loss(ce_loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            
            train_loss += ce_loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
            
        epoch_time = time.time()
            
        # Check current test accuracy of model
        test_loss, test_acc = evaluate_standard(test_loader, model)
        if test_acc > prev_test_acc:
            torch.save(model.state_dict(), best_model_path)
            prev_test_acc = test_acc
            best_epoch = epoch
        
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, 
            train_acc/train_n, test_loss, test_acc)
        
        torch.save(model.state_dict(), last_model_path)
        
        trainer_state_dict = { 'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        torch.save(trainer_state_dict, last_opt_path)
        
        
    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    # Evaluation at early stopping
    model_test = init_model(args).cuda()
    model_test.load_state_dict(torch.load(best_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    test_time = time.time()
    logger.info('Best Epoch \t Test Loss \t Test Acc \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f', best_epoch, test_loss, test_acc,
                (test_time - start_test_time)/60)
    
    # Evaluation at last model
    model_test.load_state_dict(torch.load(last_model_path))
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    test_time = time.time()
    logger.info('Last Epoch \t Test Loss \t Test Acc \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f', epoch, test_loss, test_acc, 
                (test_time - start_test_time)/60)
    
if __name__ == "__main__":
    main()
