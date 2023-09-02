# Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning


## Introduction

This is the official implementation of the paper "[Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning](https://arxiv.org/abs/2210.01035)" on [Video Swin Transformer](https://arxiv.org/abs/2106.13230).

![framework](figures/Hourglass_swin_framework.png)
![framework](figures/TokenClusterReconstruct_Details.png)

## Results 

### Kinetics 400

| Method        | $\alpha$ | t $\times$ h $\times$ w   | GFLOPs | FPS  | Acc@1 | Acc@5 | config                                                       |
| ------------- | -------- | ------------------------- | ------ | ---- | ----- | ----- | ------------------------------------------------------------ |
| Swin-L        | -        | 8 $\times$ 12 $\times$ 12 | 2107   | 1.10 | 84.7  | 96.6  | [config](configs/recognition/swin/swin_large_384_patch244_window81212_kinetics400_22k.py) |
| Swin-L + Ours | 10       | 8 $\times$ 6 $\times$ 6   | 1662   | 1.66 | 84.0  | 96.3  | [config](configs/recognition/swin/hourglass_swin_large_384_patch244_window81212_kinetics400_22k.py) |

### Kinetics 600

| Method        | $\alpha$ | t $\times$ h $\times$ w   | GFLOPs | FPS  | Acc@1 | Acc@5 | config                                                       |
| ------------- | -------- | ------------------------- | ------ | ---- | ----- | ----- | ------------------------------------------------------------ |
| Swin-L        | -        | 8 $\times$ 12 $\times$ 12 | 2107   | 1.10 | 86.1  | 97.3  | [config](configs/recognition/swin/swin_large_384_patch244_window81212_kinetics600_22k.py) |
| Swin-L + Ours | 10       | 8 $\times$ 6 $\times$ 6   | 1824   | 1.53 | 85.6  | 97.1  | [config](configs/recognition/swin/hourglass_swin_large_384_patch244_window81212_kinetics600_22k.py) |



## Usage

###  Installation

Please refer to [install.md](docs/install.md) for installation.

We also provide docker file [cuda10.1](docker/docker_10.1) ([image url](https://hub.docker.com/layers/ninja0/mmdet/pytorch1.7.1-py37-cuda10.1-openmpi-mmcv1.3.3-apex-timm/images/sha256-06d745934cb255e7fdf4fa55c47b192c81107414dfb3d0bc87481ace50faf90b?context=repo)) and [cuda11.0](docker/docker_11.0) ([image url](https://hub.docker.com/layers/ninja0/mmdet/pytorch1.7.1-py37-cuda11.0-openmpi-mmcv1.3.3-apex-timm/images/sha256-79ec3ec5796ca154a66d85c50af5fa870fcbc48357c35ee8b612519512f92828?context=repo)) for convenient usage.

###  Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.
The supported datasets are listed in [supported_datasets.md](docs/supported_datasets.md).

We also share our Kinetics-400 annotation file [k400_val](https://github.com/SwinTransformer/storage/releases/download/v1.0.6/k400_val.txt), [k400_train](https://github.com/SwinTransformer/storage/releases/download/v1.0.6/k400_train.txt) for better comparison.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> --eval top_k_accuracy

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM> --eval top_k_accuracy
```

## Citation
If you find this project useful in your research, please consider cite:

```BibTex
@article{liang2022expediting,
	author    = {Liang, Weicong and Yuan, Yuhui and Ding, Henghui and Luo, Xiao and Lin, Weihong and Jia, Ding and Zhang, Zheng and Zhang, Chao and Hu, Han},
	title     = {Expediting large-scale vision transformer for dense prediction without fine-tuning},
	journal   = {arXiv preprint arXiv:2210.01035},
	year      = {2022},
}
```

```
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}

@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
