# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F
import segmentation_models_pytorch as smp


@HEADS.register_module()
class UnetPlusPlus(BaseDecodeHead):
    def __init__(self, num_classes, deep_supervision=False, **kwargs):
        super(UnetPlusPlus, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.model = smp.UnetPlusPlus(encoder_name="vgg16", classes=64)


    def forward(self, x):
        out = self.model(x)
        out = self.cls_seg(out)
        return out



