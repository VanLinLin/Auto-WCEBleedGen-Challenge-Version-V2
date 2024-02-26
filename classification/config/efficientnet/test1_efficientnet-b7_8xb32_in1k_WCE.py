_base_ = [
    '../_base_/models/efficientnet_b7.py',
    '../_base_/datasets/WCE.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py',
]
model = dict(
    head=dict(
        num_classes=2,
        topk=(1,),
    ))

train_cfg = dict(by_epoch=True, max_epochs=150, val_interval=1)
val_cfg = dict()
test_cfg = dict()

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend')
    ])

# optimizer
optim_wrapper = dict(
    optimizer=dict(_delete_=True, type='AdamW', lr=1.25e-04, eps=1e-8, betas=(0.9, 0.999)))

work_dir = './classification/work_dirs/test1_efficientnet-b7_8xb32_in1k_WCE'