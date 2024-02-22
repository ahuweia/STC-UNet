norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNet',
        depth=50,  # resnet50
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,  # 批量归一化层的配置。这里使用标准的批量归一化。
        norm_eval=False,  # 是否在训练时冻结批量归一化层的参数。
        style='pytorch',
        contract_dilation=True
    ),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,  # 输入通道数，根据骨干网络的最后一层输出确定。
        in_index=3,  # 从骨干网络中哪个阶段提取特征。这里用3表示最后一个阶段。
        channels=512,  # 解码头内部的通道数
        pool_scales=(1, 2, 3, 6),  # Pyramid池化的尺度
        dropout_ratio=0.1,  # Dropout比例，用于防止过拟合。
        num_classes=2,  # 二分类
        norm_cfg=norm_cfg,
        align_corners=False,  # 在进行上采样时是否对齐角点，与双线性插值方法有关。
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_bce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)]))
train_cfg = dict()
test_cfg = dict(mode='whole')


dataset_type = 'MyDataset'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(600, 600)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = "../datasets/"
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/labels',
        pipeline=test_pipeline))


log_config = dict(
    interval=1065,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = False

find_unused_parameters = True


# optimizer
optimizer = dict(type='Adam', lr=1e-5, betas=(0.9, 0.999))
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=True)
# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=50)
checkpoint_config = dict(
    by_epoch=True,
    save_optimizer=False,
    interval=50)
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mFscore', 'mDice'])