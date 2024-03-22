import os
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from super_image import EdsrModel, ImageLoader
from super_image.data import EvalDataset
from datasets import load_dataset
from PIL import Image
import torch

from model import ALAN
from config import AlanConfig

exp_num = 0
scale = 4
model_name = 'ALAN'

datasets = {
    'Set5': [0, 1, 2, 3, 4],
    'Set14': [0, 3, 8, 9, 12],
    'BSD100': [15, 29, 46, 48, 92],
    'Urban100': [4, 78, 16, 42, 53, 84, 98]
    }

directory = f'./results/scale_{scale}/{model_name}'

if not os.path.exists(directory):
    os.makedirs(directory)

args = f'./config/scale{scale}/exp_{exp_num}.yml'

with open(args) as yml_file:
    cfg = yaml.load(yml_file, Loader)

    # ALAN model #
    model_name = cfg['net']['type']
    number_of_stage = cfg['net']['number_of_stage_layers_in_FEM']
    batch_norm = cfg['net']['batch_norm']
    feature_channels = cfg['net']['feature_channels']
    upsampling_ACB = cfg['net']['upsampling_ACB']
    pretrained = cfg['net']['pretrained']
    model_path = cfg['net']['model_path']

    config = AlanConfig(scale=scale, n_stage=number_of_stage, n_up_acb=upsampling_ACB)
    model = ALAN(n_channels=feature_channels, config=config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint)
    model.switch_to_deploy()
    model.eval()
    del checkpoint

for dataset_name in datasets.keys():
    eval_dataset = load_dataset(f'eugenesiow/{dataset_name}', f'bicubic_x{scale}', split='validation')
    for idx in datasets[dataset_name]:
        image = Image.open(eval_dataset[idx]['lr']).convert('RGB')
        inputs = ImageLoader.load_image(image)

        with torch.no_grad():
            preds = model(inputs)

        ImageLoader.save_image(preds, f'{directory}/{dataset_name}_{idx}.png')
        ImageLoader.save_compare(inputs, preds, f'{directory}/compare_{dataset_name}_{idx}.png')