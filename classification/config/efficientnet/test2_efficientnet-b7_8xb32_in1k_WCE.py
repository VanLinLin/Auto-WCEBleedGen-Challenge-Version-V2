_base_ = [
    './test1_efficientnet-b7_8xb32_in1k_WCE.py',
]

test_dataloader = dict(
    dataset=dict(
        data_prefix='test2')
)

work_dir = './classification/work_dirs/test2_efficientnet-b7_8xb32_in1k_WCE'