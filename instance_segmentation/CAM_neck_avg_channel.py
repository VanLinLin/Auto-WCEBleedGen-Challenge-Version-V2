import argparse
import mmcv_custom
import mmdet_custom
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from pathlib import Path
from tqdm.auto import tqdm


class FeatureExtractionHook:
    def __init__(self):
        self.features = None

    def __call__(self, module, module_in, module_out):
        self.features = module_out

def init_model(config_path,
               checkpoint_path):
    cfg = Config.fromfile(config_path)
    model = init_detector(config_path, checkpoint_path, device='cuda:0')
    dataset = build_dataset(cfg.data.test)
    model.CLASSES = dataset.CLASSES
    return model


def insert_feature_extraction_hook(model):
    hook = FeatureExtractionHook()
    handle = model.neck.register_forward_hook(hook)
    return hook, handle


def generate_feature_heatmap(save_path, model, image_path, feature_hook, combine_cam_and_result):
    cam_save_path: Path = save_path / 'CAM'
    cam_save_path.mkdir(parents=True, exist_ok=True)
    predict_result_save_path = save_path / 'predict_results'
    predict_result_save_path.mkdir(parents=True, exist_ok=True)

    if Path(image_path).is_file():
        image_list = [Path(image_path)]
    else:
        image_list = list(Path(image_path).glob('*.png'))
        
    for image in tqdm(image_list):

        result = inference_detector(model, image)

        show_result_pyplot(model=model,
                           img=image,
                           result=result,
                           score_thr=0,
                           out_file=f'{predict_result_save_path}/{image.name}')

        features = feature_hook.features

        features_mean = torch.mean(features[4][0], dim=0)
        features_norm = (features_mean - features_mean.min()) / (features_mean.max() - features_mean.min())
        

        heatmap = plt.get_cmap('jet')(features_norm.cpu().numpy())[:, :, :3]
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).convert('RGB')

        img = Image.open(image).convert('RGB')

        heatmap_resized = heatmap.resize(img.size, Image.BILINEAR)

        img_array = np.array(img)
        heatmap_array = np.array(heatmap_resized)
        superimposed_img = img_array * 0.5 + heatmap_array * 0.5
        superimposed_img = np.uint8(superimposed_img)

        cv2.imwrite(f'{cam_save_path}/{image.name}', cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    if combine_cam_and_result:
        print("Start to combine predict result and cam image!")
        combine_path = save_path / 'combine_cam_and_predict_result'
        combine_path.mkdir(parents=True, exist_ok=True)

        # plt.figure(figsize=(8, 5))
        for image in tqdm(list(cam_save_path.glob('*.png'))):
            image_name = image.name
            cam_image = cam_save_path / image_name
            result_image = predict_result_save_path / image_name

            plt.subplot(121)
            plt.imshow(Image.open(result_image))
            plt.title(f'Predict result')
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(Image.open(cam_image))
            plt.title(f'Feature heatmap')
            plt.axis('off')

            plt.suptitle(f'{image.stem}')

            plt.savefig(f'{combine_path}/combine_{image_name}', bbox_inches='tight')

            plt.clf()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path',
                        default='internimage_xl_CAM_results',
                        help='the path where visualize results stored')

    parser.add_argument('--img_path',
                        default='instance_segmentation/data/WCEBleedGen_v2/instance_seg_img_test1/Images',
                        help='image folder path or single image path')
    
    parser.add_argument('--combine',
                        action='store_true',
                        help='combine the predict image and CAM image')

    return parser.parse_args()

def main(model):
    args = parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    feature_hook, handle = insert_feature_extraction_hook(model)
    
    generate_feature_heatmap(save_path, model, args.img_path, feature_hook, args.combine)

    handle.remove()

if __name__ == '__main__':
    config_file = 'instance_segmentation/configs/coco/test1_internimage_xl.py'
    checkpoint_file = 'instance_segmentation/weight/swa_internimage_xl.pth'

    model = init_model(config_file, checkpoint_file)


    main(model=model)


