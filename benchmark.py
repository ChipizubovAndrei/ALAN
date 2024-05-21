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

from utils import LoggerCSV, convert_image


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
        inputs = convert_image(inputs, source='[0, 1]', target='imagenet-norm')

        with torch.no_grad():
            preds = model(inputs)

        preds = convert_image(preds, source='[-1, 1]', target='[0, 1]')
        metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

        epoch_psnr.update(metrics['psnr'], len(inputs))
        epoch_ssim.update(metrics['ssim'], len(inputs))

    print(f"{dataset}:")
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

exp_num = 6
scale = 4
args = f'./config/scale{scale}/exp_{exp_num}.yml'

with open(args) as yml_file:
    cfg = yaml.load(yml_file, Loader)

    # ALAN model #
    model_name = cfg['net']['type']
    pretrained = cfg['net']['pretrained']
    model_path = cfg['net']['model_path']

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

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params:,} trainable parameters')

# benchmarks = ["Set5", "Set14", "BSD100", "Urban100"]
benchmarks = ["Set5", "BSD100", "Urban100"]
# benchmarks = ["Set5"]
for data in benchmarks:
    benchmark(data, scale)
