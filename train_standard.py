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

from preactresnet import PreActResNet18
from utils import (upper_limit, lower_limit, cifar10_mean, cifar10_std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard)
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
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--activation', default='relu', choices=['relu', 'swish', 'softplus', 
        'tanh', 'sigmoid'], help='Activation function')
    parser.add_argument('--out-dir', default='standard', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--conv-type', default='standard', type=str, choices=['standard', 'bcop', 
        'skew'], help='standard, skew symmetric or bcop convolution')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], 
                        help='dataset to use for training')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    return parser.parse_args()


def main():
    args = get_args()

    args.out_dir = args.out_dir + '_' + str(args.dataset) + '_' + args.activation + '_' + str(args.conv_type)
    os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(args.out_dir, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.dataset)
    std = cifar10_std
    if args.dataset == 'cifar10':
        num_classes = 10    
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        raise Exception('Unknown dataset')

    model = PreActResNet18(args.activation, args.conv_type, num_classes).cuda()
    model.train()

    if args.conv_type in ['standard', 'skew']:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    elif args.conv_type in ['bcop']:
        conv_params = []
        not_conv_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'conv' in name:
                    conv_params.append(param)
                else:
                    not_conv_params.append(param)

        opt = torch.optim.SGD([
                                {'params': conv_params, 'weight_decay': 0.},
                                {'params': not_conv_params, 'weight_decay': args.weight_decay}
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
        
    # Training
    l_weight = model.linear.weight
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
        last_state_dict = copy.deepcopy(model.state_dict())
            
        # Check current test accuracy of model
        test_loss, test_acc = evaluate_standard(test_loader, model)
        if test_acc > prev_test_acc:
            model_path = os.path.join(args.out_dir, 'best.pth')
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save(best_state_dict, model_path)
            prev_test_acc = test_acc
            best_epoch = epoch
        
        lr = scheduler.get_last_lr()[0]
        logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, 
            train_acc/train_n, test_loss, test_acc)
        
        model_path = os.path.join(args.out_dir, 'last.pth')
        torch.save(last_state_dict, model_path)
        
        trainer_state_dict = { 'epoch': epoch, 'optimizer_state_dict': opt.state_dict()}
        opt_path = os.path.join(args.out_dir, 'last_opt.pth')
        torch.save(trainer_state_dict, opt_path)
        
        if args.conv_type=='skew':
            sigma_array = test_real_sn(model)
            s_min, s_mean, s_max = sigma_array.min(), sigma_array.mean(), sigma_array.max()
            logger.info('Real sigma statistics: %.4f \t %.4f \t %.4f', s_min, s_mean, s_max)            

        
    train_time = time.time()

    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    # Evaluation at early stopping
    model_test = PreActResNet18(args.activation, args.conv_type, num_classes).cuda()
    model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    start_test_time = time.time()
    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    test_time = time.time()
    logger.info('Best Epoch \t Test Loss \t Test Acc \t Test Time')
    logger.info('%d \t %.4f \t %.4f \t %.4f', best_epoch, test_loss, test_acc,
                (test_time - start_test_time)/60)
    
    # Evaluation at last model
    model_test = PreActResNet18(args.activation, args.conv_type, num_classes).cuda()
    model_test.load_state_dict(last_state_dict)
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
