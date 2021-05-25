# Skew Orthogonal Convolutions

SOC is a convolution layer that has an Orthogonal Jacobian matrix and achieves improved standard and provably robust accuracy over the prior works.

## Prerequisites

+ Python 3.7+, Pytorch 1.6+
+ A recent NVIDIA GPU

## How to run?

+ Run python train_robust.py --conv-type skew  --block-size BLOCK_SIZE --dataset DATASET_NAME

## Demonstration

For the images with label water jug, when feature[1378] (visually identified as 'water jug handle') is less than 0.089335, error rate increases to 100.0% from 50.0%, i.e an increase of 50.0% in the failure rate.

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
