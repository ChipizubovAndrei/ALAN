import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import os

from datasets import load_dataset
from model import ALAN
from super_image import TrainingArguments, EdsrConfig
from trainer import Trainer

from datasets import load_dataset
from super_image.data import EvalDataset, TrainDataset, augment_five_crop

from torch.optim import Adam, Adagrad, SGD
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR, StepLR
from torch import nn
import torch

from config import AlanConfig

from utils import TruncatedVGG19

os.chdir(os.path.dirname(__file__))

exp_num = 4
scale = 4
args = f'./config/scale{scale}/exp_{exp_num}.yml'

with open(args) as yml_file:
    cfg = yaml.load(yml_file, Loader)

    output_dir = cfg['logging']['model_save_dir']

    epoch = 0
    num_train_epochs = cfg['train']['epoch_size']

    dataset_train = cfg['dataset']['train']['name']
    dataset_valid = cfg['dataset']['valid']['name']

    batch_size = cfg['train_batcher']['batch']
    patch_size = cfg['preproc']['patch_size']   #Размер изображений в батче

# --- Model --- #
    model_name = cfg['net']['type']
    number_of_stage = cfg['net']['number_of_stage_layers_in_FEM']
    batch_norm = cfg['net']['batch_norm']
    feature_channels = cfg['net']['feature_channels']
    upsampling_type = cfg['net']['upsampling']['type']
    upsampling_ACB = cfg['net']['upsampling']['n_ACB']
    pretrained = cfg['net']['pretrained']
    if pretrained:
        model_path = cfg['net']['model_path']
    else:
        model_path = ''

    config = AlanConfig(
        scale=scale, n_stage=number_of_stage, 
        n_up_acb=upsampling_ACB, upsampling_type=upsampling_type, 
        batch_norm=batch_norm)
    model = ALAN(n_channels=feature_channels, config=config, model_name=model_name)

    if torch.cuda.is_available():
        model = model.cuda()

# --- Optimizer --- #
    opt_name = cfg['opt']['type']
    lr = cfg['opt']['lr']

    if opt_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)

# --- Scheduler --- #
    scheduler_name = cfg['lr_scheduler']['type']
    step_size = cfg['lr_scheduler']['step_size']
    gamma = cfg['lr_scheduler']['gamma']

    if scheduler_name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# --- Loss --- #
    loss_name = cfg['loss']['type']
    if loss_name == 'SmoothL1':
        loss = nn.SmoothL1Loss()
    elif loss_name == 'ContentLoss':
        loss = nn.MSELoss()
        vgg19_i = cfg['loss']['vgg19_i']  # the index i in the definition for VGG loss; see paper or models.py
        vgg19_j = cfg['loss']['vgg19_j']  # the index j in the definition for VGG loss; see paper or models.py
        truncated_vgg19 = TruncatedVGG19(vgg19_i, vgg19_j)
        truncated_vgg19.eval()
            
    if torch.cuda.is_available():
        truncated_vgg19 = truncated_vgg19.cuda()
        loss = loss.cuda() 

# --- Checkpoint --- #
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        del checkpoint

#-------------------------------------------------------------------------------#
# download and augment the data with the five_crop method
augmented_dataset = load_dataset(f'eugenesiow/{dataset_train}', f'bicubic_x{scale}', split='train')\
                    .map(augment_five_crop, batched=True, batch_size=batch_size, desc="Augmenting Dataset")
                    
# prepare the train dataset for loading PyTorch DataLoader
train_dataset = TrainDataset(augmented_dataset, patch_size=patch_size)                                
eval_dataset = EvalDataset(load_dataset(f'eugenesiow/{dataset_valid}', f'bicubic_x{scale}', split='validation'))

training_args = TrainingArguments(
    output_dir=output_dir,   # output directory
    num_train_epochs=num_train_epochs,                  # total number of training epochs
    per_device_train_batch_size=batch_size,
    save_steps=100
)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,           # evaluation dataset
    optimizer=optimizer,
    criterion=loss,
    scheduler=scheduler,
    epoch=epoch
)

trainer.train()
