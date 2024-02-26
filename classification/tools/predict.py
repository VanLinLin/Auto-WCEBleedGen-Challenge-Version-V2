import argparse
from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain.visualization import UniversalVisualizer
from mmcv.image import imread

from pathlib import Path


def init_model():
    config_path = 'classification/config/efficientnet/test1_efficientnet-b7_8xb32_in1k_WCE.py'
    checkpoint_path = 'classification/weight/swa_efficient_b7_album_102e_last_5_checkpoint_classification.pth'

    inference = ImageClassificationInferencer(model=config_path,
                                              pretrained=checkpoint_path,
                                              device='cuda:0',
                                              classes=('bleeding', 'non-bleeding'))
    
    return inference


def inference_result(model, img_path):
    image_path = Path(img_path)
    save_path = Path('./inference_pipeline')
    save_path.mkdir(parents=True,exist_ok=True)
    bleeding_folder = save_path / 'bleeding'
    non_bleeding_folder = save_path / 'non-bleeding'
    
    bleeding_folder.mkdir(parents=True, exist_ok=True)
    non_bleeding_folder.mkdir(parents=True, exist_ok=True)
    
    visualizer = UniversalVisualizer()


    if image_path.is_file():
        result = model(img_path, return_datasamples=True)
        if model.classes[result[0].pred_label.tolist()[0]] == 'bleeding':
            visualizer.visualize_cls(
                image=imread(str(image_path), channel_order='rgb'),
                data_sample=result[0],
                classes=model.classes,
                out_file=bleeding_folder / (str(image_path.name))
            )
        else:
            visualizer.visualize_cls(
                image=imread(str(image_path), channel_order='rgb'),
                data_sample=result[0],
                classes=model.classes,
                out_file=non_bleeding_folder / (str(image_path.name))
            )
    else:
        for img in image_path.glob('*'):
            result = model(img, return_datasamples=True)
            if model.classes[result[0].pred_label.tolist()[0]] == 'bleeding':
                visualizer.visualize_cls(
                    image=imread(str(img), channel_order='rgb'),
                    data_sample=result[0],
                    classes=model.classes,
                    out_file=bleeding_folder / (str(img.name))
                )
            else:
                visualizer.visualize_cls(
                    image=imread(str(img), channel_order='rgb'),
                    data_sample=result[0],
                    classes=model.classes,
                    out_file=non_bleeding_folder / (str(img.name))
                )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path',
                        default='classification/data/WCEBleedGen_v2/test1/bleeding',
                        help='image path or images folder')

    return parser.parse_args()

def main():
    args = parse_args()
    
    model = init_model()

    inference_result(model, args.image_path)


if __name__ == '__main__':
    main()

