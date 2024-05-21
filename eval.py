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
from utils import LoggerCSV, convert_image

from models import SRResNet

os.chdir(os.path.dirname(__file__))

exp_num = 6
scale = 4
model_name = 'ALANSR'

datasets = {
    'Set5': [0, 1, 2, 3, 4],
    # 'Set14': [0, 3, 8, 9, 12],
    # 'BSD100': [15, 29, 46, 48, 92],
    # 'Urban100': [4, 78, 16, 42, 53, 84, 98]
    }

directory = f'./results/scale_{scale}/{model_name}'

if not os.path.exists(directory):
    os.makedirs(directory)

args = f'./config/scale{scale}/exp_{exp_num}.yml'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(args) as yml_file:
    cfg = yaml.load(yml_file, Loader)

    # ALAN model #
    model_name = cfg['net']['type']
    number_of_stage = cfg['net']['number_of_stage_layers_in_FEM']
    batch_norm = cfg['net']['batch_norm']
    feature_channels = cfg['net']['feature_channels']
    pretrained = cfg['net']['pretrained']
    model_path = cfg['net']['model_path']

    # model = SRResNet(large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4)

    srgan_checkpoint = "./results/models/alansr/checkpoint_srgan.pth.tar"
    srresnet_checkpoint = "./results/models/alansr/checkpoint_srresnet.pth.tar"

    # Load model, either the SRResNet or the SRGAN
    # srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    # srresnet.eval()
    # model = srresnet
    srgan_generator = torch.load(srgan_checkpoint)['generator']

    model = srgan_generator
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    del checkpoint

for dataset_name in datasets.keys():
    eval_dataset = load_dataset(f'eugenesiow/{dataset_name}', f'bicubic_x{scale}', split='validation')
    for idx in datasets[dataset_name]:
        image = Image.open(eval_dataset[idx]['lr']).convert('RGB')
        inputs = ImageLoader.load_image(image)
        inputs_norm = inputs
        inputs_norm = inputs_norm.cuda()
        inputs_norm = convert_image(inputs_norm, source='[0, 1]', target='imagenet-norm')

        with torch.no_grad():
            preds = model(inputs_norm)
        
        preds = convert_image(preds, source='[-1, 1]', target='[0, 1]')
        preds = preds.cpu()
        ImageLoader.save_image(preds, f'{directory}/{dataset_name}_{idx}.png')
        ImageLoader.save_compare(inputs, preds, f'{directory}/compare_{dataset_name}_{idx}.png')