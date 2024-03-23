import os
import time
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import *
from super_image.trainer_utils import EvalPrediction

from datasets import load_dataset
from super_image.utils.metrics import AverageMeter, compute_metrics
from super_image.data import EvalDataset 

from config import AlanConfig
from model import ALAN

os.chdir(os.path.dirname(__file__))

def get_eval_dataloader(eval_dataset):
    return DataLoader(
        dataset=eval_dataset,
        batch_size=1,
    )

def benchmark(dataset, scale):
    eval_dataset = EvalDataset(load_dataset(f'eugenesiow/{dataset}', f'bicubic_x{scale}', split='validation'))
    eval_dataloader = get_eval_dataloader(eval_dataset)
    epoch_psnr = AverageMeter()
    epoch_ssim = AverageMeter()

    start_time = time.time()
    for data in eval_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs)

        metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

        epoch_psnr.update(metrics['psnr'], len(inputs))
        epoch_ssim.update(metrics['ssim'], len(inputs))

    print(f"{dataset}:")
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_num = 3
scale = 4
args = f'./config/scale{scale}/exp_{exp_num}.yml'

with open(args) as yml_file:
    cfg = yaml.load(yml_file, Loader)

    # ALAN model #
    model_name = cfg['net']['type']
    number_of_stage = cfg['net']['number_of_stage_layers_in_FEM']
    batch_norm = cfg['net']['batch_norm']
    feature_channels = cfg['net']['feature_channels']
    upsampling_type = cfg['net']['upsampling']['type']
    upsampling_ACB = cfg['net']['upsampling']['n_ACB']
    pretrained = cfg['net']['pretrained']
    model_path = cfg['net']['model_path']

    config = AlanConfig(
        scale=scale, n_stage=number_of_stage, 
        n_up_acb=upsampling_ACB, upsampling_type=upsampling_type, 
        batch_norm=batch_norm)
    model = ALAN(n_channels=feature_channels, config=config, model_name=model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device)
    model.switch_to_deploy()
    model.eval()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params:,} trainable parameters')

# benchmarks = ["Set5", "Set14", "BSD100", "Urban100"]
benchmarks = ["Set5"]
for data in benchmarks:
    benchmark(data, scale)
