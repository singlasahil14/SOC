# Skew Orthogonal Convolutions

+ **Skew Orthogonal Convolution (SOC)** is a convolution layer that has an Orthogonal Jacobian matrix and achieves state-of-the-art standard and provably robust accuracy compared to the other orthogonal convolutions. 
+ **Last Layer normalization (LLN)** leads to improved performance when the number of classes is large.
+ **Certificate Regularization (CR)** leads to significantly improved robustness certificates.
+ **Householder Activations (HH)** improves the performance for deeper networks.

## Prerequisites

+ Python 3.7+, Pytorch 1.6+
+ A recent NVIDIA GPU

## How to train 1-Lipschitz Convnets?

```python train_robust.py --conv-layer CONV_LAYER --activation ACTIVATION_NAME --num-blocks BLOCK_SIZE --dataset DATASET_NAME --gamma GAMMA```
+ CONV_LAYER can be bcop/cayley/soc
+ ACTIVATION_NAME can be maxmin/hh1/hh2. hh1 is the householder activation of order 1, hh2 is the householder activation of order 2. Both are illustrated in Figures 1 and 2 in the paper titled "Improved deterministic l2 robustness on CIFAR-10 and CIFAR-100"
+ GAMMA is the certificate regularization coefficient
+ Use the flag ```--lln``` to activate last layer normalization
+ DATASET_NAME can be cifar10/cifar100.

## How to train Standard Convnets using Orthogonal Convolutions?
```python train_standard.py --conv-layer CONV_LAYER --model-name MODEL_NAME --dataset DATASET_NAME```
+ DATASET_NAME can be cifar10/cifar100. CONV_LAYER can be standard/bcop/cayley/soc, MODEL_NAME can be resnet18/resnet34/resnet50/resnet101/resnet152

## Demonstration

![demo](./figures/SOC_demo.png)

## Citations
If you find this repository useful for your research, please cite:

```
@inproceedings{singlafeiziSOC2021,
  title={Skew Orthogonal Convolutions},
  author={Sahil Singla and Soheil Feizi},
  booktitle={ICML},
  year={2021}
}

@inproceedings{singla2022improved,
title={Improved deterministic l2 robustness on {CIFAR}-10 and {CIFAR}-100},
author={Sahil Singla and Surbhi Singla and Soheil Feizi},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=tD7eCtaSkR}
}
```

