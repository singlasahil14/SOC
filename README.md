# Skew Orthogonal Convolutions, Last layer normalization, Certificate Regularization and Householder Activations

SOC is a convolution layer that has an Orthogonal Jacobian matrix and achieves state-of-the-art standard and provably robust accuracy compared to the prior works on orthogonal convolutions. Last Layer normalization, Certificate Regularization and Householder Activations are additional tricks that lead to improved performance.

## Prerequisites

+ Python 3.7+, Pytorch 1.6+
+ A recent NVIDIA GPU

## How to run?

+ Run ```python train_robust.py --conv-layer CONV_LAYER  --num-blocks BLOCK_SIZE --dataset DATASET_NAME```
+ Here, DATASET_NAME can be either cifar10 or cifar100. CONV_LAYER can be either soc, bcop, cayley.

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

@inproceedings{
singla2022improved,
title={Improved deterministic l2 robustness on {CIFAR}-10 and {CIFAR}-100},
author={Sahil Singla and Surbhi Singla and Soheil Feizi},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=tD7eCtaSkR}
}
```

