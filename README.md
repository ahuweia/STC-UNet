# STC-UNet

This study is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation.git).

## 1 Installation

Important versions

```
python==3.8
pytorch==1.10.0 
torchvision==0.11.0 
torchaudio==0.10.0 
cudatoolkit=11.3
mmcv-full==1.7.0
timm==0.4.12 
segmentation-models-pytorch==0.2.0 
yapf==0.40.1
```

**Step 1.** Create a conda environment and activate it.

```
conda create --name py38 python=3.8 -y
conda activate py38
```

**Step 2.** Install [PyTorch](https://pytorch.org/) using following command.

```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**Step 3.** Install MMCV using following command.

```
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

**Step 4.** Install other packages.

```
pip install mmengine tensorboard timm==0.4.12 segmentation-models-pytorch==0.2.0 opencv-python einops yapf==0.40.1
```

**Step 5.** Clone this repository.

```
git clone https://github.com/ahuweia/STC-UNet.git
cd STC-UNet
pip install -e .
```

## 2 Datasets

This study uses the [KiTS19](https://github.com/neheller/kits19) dataset for related experiments.

## 3 Training

```
python tools/train.py my_config/STC-UNet.py
```

## 4 Inference

```
python tools/test.py "the path of your test data" "my_config/STC-UNet.py" "path/to/checkpoint.pth"
```