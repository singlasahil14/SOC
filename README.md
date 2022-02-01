# Skew Orthogonal Convolutions

+ **Skew Orthogonal Convolution (SOC)** is a convolution layer that has an Orthogonal Jacobian matrix and achieves state-of-the-art standard and provably robust accuracy compared to the other orthogonal convolutions. 
+ **Last Layer normalization (LLN)** leads to improved performance when the number of classes is large.
+ **Certificate Regularization (CR)** leads to significantly improved robustness certificates.
+ **Householder Activations (HH)** improve the performance for deeper networks.

## Prerequisites

+ Python 3.7 or 3.8, Pytorch 1.8+
+ NVIDIA Apex. Can be installed using ```conda install -c conda-forge nvidia-apex```
+ A recent NVIDIA GPU

## How to train 1-Lipschitz Convnets?

```python train_robust.py --conv-layer CONV --activation ACT --num-blocks BLOCKS --dataset DATASET --gamma GAMMA```
+ CONV: bcop, cayley, soc
+ ACT: maxmin, hh1, hh2. (hh1, hh2 are householder activations of order 1, 2 respectively; both illustrated below).
+ BLOCKS: 1, 2, 3, 4, 5, 6, 7, 8
+ GAMMA: certificate regularization coefficient
+ Use ```--lln``` to enable last layer normalization
+ DATASET: cifar10/cifar100.

## How to train Standard Convnets using Orthogonal Convolutions?
```python train_standard.py --conv-layer CONV --model-name MODEL --dataset DATASET```
+ CONV: standard, bcop, cayley, soc
+ MODEL: resnet18, resnet34, resnet50, resnet101, resnet152
+ DATASET: cifar10, cifar100

## Demonstration of Skew Orthogonal Convolutions

![demo](./figures/SOC_demo.png)

## Demonstration of Householder Activations
+ Illustration of the hh1 activation function
![demo](./figures/hh1_demo.png)
+ Illustration of the hh2 activation function (right)
![demo](./figures/hh2_demo.png)


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
  year={2022}
}
```

