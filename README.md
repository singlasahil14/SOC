# Skew Orthogonal Convolutions

SOC is a convolution layer that has an Orthogonal Jacobian matrix and achieves improved standard and provably robust accuracy over the prior works.

## Prerequisites

+ Python 3.7+, Pytorch 1.6+
+ A recent NVIDIA GPU

## How to run?

+ Run ```python train_robust.py --conv-type skew  --block-size BLOCK_SIZE --dataset DATASET_NAME```

## Demonstration

![demo](./figures/SOC_demo.png)

## Citation

```
@inproceedings{singlafeiziSOC2021,
  title={Skew Orthogonal Convolutions},
  author={Sahil Singla and Soheil Feizi},
  booktitle={ICML},
  year={2021}
}
```
