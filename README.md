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

This repo contains the experiment code for [Swin3D](https://github.com/microsoft/Swin3D)


## Overview
- [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results and models](#results-and-models)
- [Citation](#citation)

## Environment
  1. Install dependencies

            pip install -r requirements.txt
      
  2. Refer to this [repo](https://github.com/microsoft/swin3d) to compile the operation of swin3d

            git clone https://github.com/microsoft/Swin3D
            cd Swin3D
            python setup.py install


If you have problems installing the package, you can use the docker we provide:

      docker pull yukichiii/torch112_cu113:swin3d

## Data Preparation

### ScanNet Segmentation Data
Please refer to https://github.com/dvlab-research/PointGroup for the ScanNetv2 preprocessing. Then change the data_root entry in the yaml files in  `SemanticSeg/config/scannetv2`.

### S3DIS Segmentation Data
Please refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch for S3DIS preprocessing. Then modify the data_root entry in the yaml files in `SemanticSeg/config/s3dis`.

### ScanNet 3D Detection Data

### S3DIS 3D Detection Data

## Training
### ScanNet Segmentation
Change the work directory to SemanticSeg

      cd SemanticSeg

To train model on ScanNet Segmentation Task with Swin3D-S or Swin3D-L from scratch:

      python train.py --config config/scannetv2/swin3D_RGBN_S.yaml
      or
      python train.py --config config/scannetv2/swin3D_RGBN_L.yaml

To finetune the model pretrained on Structured3D, you can download the pretrained model with cRSE(XYZ,RGB,Norm) [here](https://github.com/microsoft/Swin3D#pretrained-models), and run:

      python train.py --config config/scannetv2/swin3D_RGBN_S.yaml args.weight PATH_TO_PRETRAINED_SWIN3D_RGBN_S
      or
      python train.py --config config/scannetv2/swin3D_RGBN_L.yaml args.weight PATH_TO_PRETRAINED_SWIN3D_RGBN_L

### S3DIS Segmentation
Change the work directory to SemanticSeg

      cd SemanticSeg

To train model on S3DIS Area5 Segmentation with Swin3D-S or Swin3D-L from scratch:

      python train.py --config config/s3dis/swin3D_RGB_S.yaml
      or
      python train.py --config config/s3dis/swin3D_RGB_L.yaml

To finetune the model pretrained on Structured3D, you can download the pretrained model with cRSE(XYZ,RGB) [here](https://github.com/microsoft/Swin3D#pretrained-models), and run:

      python train.py --config config/s3dis/swin3D_RGB_S.yaml args.weight PATH_TO_PRETRAINED_SWIN3D_RGB_S
      or
      python train.py --config config/s3dis/swin3D_RGB_L.yaml args.weight PATH_TO_PRETRAINED_SWIN3D_RGB_L

## Evaluation
To forward Swin3D with given checkpoint with TTA(Test Time Augmentation, we random rotate the input scan and vote the result), you can download the model [below](#results-and-models) and run:

ScanNet Segmentation

      python test.py --config config/scannetv2/swin3D_RGBN_S.yaml --vote_num 12 args.weight PATH_TO_CKPT
      or
      python test.py --config config/scannetv2/swin3D_RGBN_L.yaml --vote_num 12 args.weight PATH_TO_CKPT

S3DIS Area5 Segmentation

      python test.py --config config/s3dis/swin3D_RGB_S.yaml --vote_num 12 args.weight PATH_TO_CKPT
      or
      python test.py --config config/s3dis/swin3D_RGB_L.yaml --vote_num 12 args.weight PATH_TO_CKPT

For faster forward, you can change the `vote_num` to 1.

## Results and models

### ScanNet Segmentation

|          | Pretrained | mIoU(Val)  | mIoU(Test) |                                            Model                                            |                                           Train                                           |                                           Eval                                            |
| :------- | :--------: | :--------: | :--------: | :-----------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| Swin3D-S |  &cross;   |    75.2    |     -      |                                            model                                            |                                            log                                            |                                            log                                            |
| Swin3D-S |  &check;   | 75.6(76.8) |     -      | [model](https://drive.google.com/file/d/1ttObwwJMrW2_9gd3xgrvpzddY0khIAbd/view?usp=sharing) | [log](https://drive.google.com/file/d/13Wqi3fY0WY9hcI9HLnsgX_HOe9nFHgYz/view?usp=sharing) | [log](https://drive.google.com/file/d/1RcmkyojYuEmBxkoW9Rnz-eTdBLiglJlJ/view?usp=sharing) |
| Swin3D-L |  &check;   | 76.4(77.5) |    77.9    | [model](https://drive.google.com/file/d/1dtpWflstuH7wZA925PN10YS_JEUtWDMp/view?usp=sharing) | [log](https://drive.google.com/file/d/1niVSJa6afSBcFSpvgtr2gpoDZLhm1ZgV/view?usp=sharing) | [log](https://drive.google.com/file/d/1JgNfh6_ZRxu021WhptZdDvtyXZ3lb82s/view?usp=sharing) |


### S3DIS Segmentation

|          | Pretrained | Area 5 mIoU | 6-fold mIoU |                                            Model                                            |                                           Train                                           |                                           Eval                                            |
| :------- | :--------: | :---------: | :---------: | :-----------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
| Swin3D-S |  &cross;   |    72.5     |    76.9     |                                            model                                            |                                            log                                            |                                            log                                            |
| Swin3D-S |  &check;   |    73.0     |    78.2     | [model](https://drive.google.com/file/d/1zt07ikG7haT-nyW1wW38DSpbO1TD86CA/view?usp=sharing) | [log](https://drive.google.com/file/d/1-RRvKCigLBtdDR22y_2ftbGZrzhDaF3z/view?usp=sharing) | [log](https://drive.google.com/file/d/1Mgd1GFsYGyE_AUL0V701Jn66dDvrYq1_/view?usp=sharing) |
| Swin3D-L |  &check;   |    74.5     |    79.8     | [model](https://drive.google.com/file/d/12QOZDcNUpSwXWhMlVY7nm5lnk4jwwyfs/view?usp=sharing) | [log](https://drive.google.com/file/d/1hm3MPJ2ZdmduX6Dncc1HlO_MwUE8-Tl4/view?usp=sharing) | [log](https://drive.google.com/file/d/1zDEO3-kHMRLTK2RvDVdV_FrBt6cSZ3dw/view?usp=sharing) |
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
