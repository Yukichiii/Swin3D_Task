# Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/semantic-segmentation-on-scannet)](https://paperswithcode.com/sota/semantic-segmentation-on-scannet?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/semantic-segmentation-on-s3dis-area5)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis-area5?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/semantic-segmentation-on-s3dis)](https://paperswithcode.com/sota/semantic-segmentation-on-s3dis?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=swin3d-a-pretrained-transformer-backbone-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/swin3d-a-pretrained-transformer-backbone-for/3d-object-detection-on-s3dis)](https://paperswithcode.com/sota/3d-object-detection-on-s3dis?p=swin3d-a-pretrained-transformer-backbone-for)

## Updates

***27/04/2023***

Initial commits:

1. The supported code and models for Semantic Segmentation on ScanNet and S3DIS are provided.

## Introduction

This repo contains the experiment code for [Swin3D](https://github.com/microsoft/Swin3D/actions)


## Overview
- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Pretrained Models](#pretrained-models)
- [Quick Start](#quick-start)
- [Results and models](#results-and-models)
- [Citation](#citation)

## Data Preparation

## Environment

### ScanNet Segmentation Data

### S3DIS Segmentation Data

### ScanNet 3D Detection Data

### S3DIS 3D Detection Data

## Results and models

### ScanNet Segmentation

|          | Pretrained | mIoU(Val) | mIoU(Test) |   Model   |   Log   |
| :------- | :--------: | :-------: | :--------: | :-------: | :-----: |
| Swin3D-S |  &cross;   |   75.2    |     -      | [model]() | [log]() |
| Swin3D-S |  &check;   |   75.7    |     -      | [model]() | [log]() |
| Swin3D-L |  &check;   |   77.5    |    77.9    | [model]() | [log]() |

### S3DIS Segmentation

|          | Pretrained | Area 5 mIoU | 6-fold mIoU |   Model   |   Log   |
| :------- | :--------: | :---------: | :---------: | :-------: | :-----: |
| Swin3D-S |  &cross;   |    72.5     |    76.9     | [model]() | [log]() |
| Swin3D-S |  &check;   |    73.0     |    78.2     | [model]() | [log]() |
| Swin3D-L |  &check;   |    74.5     |    79.8     | [model]() | [log]() |

### ScanNet 3D Detection

|                    | Pretrained | mAP@0.25 | mAP@0.50 | Model |  Log  |
| :----------------- | :--------: | :------: | :------: | :---: | :---: |
| Swin3D-S+FCAF3D    |  &check;   |   74.2   |   59.5   | model |  log  |
| Swin3D-L+FCAF3D    |  &check;   |   74.2   |   58.6   | model |  log  |
| Swin3D-S+CAGroup3D |  &check;   |   76.4   |   62.7   | model |  log  |
| Swin3D-L+CAGroup3D |  &check;   |   76.4   |   63.2   | model |  log  |

### S3DIS 3D Detection

|                 | Pretrained | mAP@0.25 | mAP@0.50 | Model |  Log  |
| :-------------- | :--------: | :------: | :------: | :---: | :---: |
| Swin3D-S+FCAF3D |  &check;   |   69.9   |   50.2   | model |  log  |
| Swin3D-L+FCAF3D |  &check;   |   72.1   |   54.0   | model |  log  |

## Citation

If you find Swin3D useful to your research, please cite our work:

```
@misc{yang2023swin3d,
      title={Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding}, 
      author={Yu-Qi Yang and Yu-Xiao Guo and Jian-Yu Xiong and Yang Liu and Hao Pan and Peng-Shuai Wang and Xin Tong and Baining Guo},
      year={2023},
      eprint={2304.06906},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
