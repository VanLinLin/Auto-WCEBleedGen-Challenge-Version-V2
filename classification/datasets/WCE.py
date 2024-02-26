bgr_mean = [103.53, 116.28, 123.675]
albu_train_transforms = [
    dict(type='RandomRotate90',
         p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1.0),
            dict(
                type='CLAHE',
                p=1.0
            )
        ],
        p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='Albu',
        transforms=albu_train_transforms),
    # dict(type='NumpyToPIL', to_rgb=True),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='NumpyToPIL', to_rgb=True),
    # dict(type='CenterCrop', crop_size=640),
    dict(type='PackInputs')
]

dataset_type = 'CustomDataset'
data_root = 'data/WCEBleedGen_v2'
metainfo = {
    'classes': ('bleeding', 'non-bleeding')
}
train_dataloader = dict(
    batch_size=30,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=10,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = [
  dict(type='Accuracy', topk=(1)),
  dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score'])
]
test_evaluator = val_evaluator