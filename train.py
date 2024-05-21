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

exp_num = 6
scale = 4
args = f'./config/scale{scale}/exp_{exp_num}.yml'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    srgan_checkpoint = "./results/models/alansr/checkpoint_srgan.pth.tar"
    srresnet_checkpoint = "./results/models/alansr/checkpoint_srresnet.pth.tar"

    # Load model, either the SRResNet or the SRGAN
    # srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
    # srresnet.eval()
    # model = srresnet
    srgan_generator = torch.load(srgan_checkpoint)['generator']
    for param in srgan_generator.parameters():
        param.requires_grad = False

    for param in srgan_generator.net.conv_block3.parameters():
        param.requires_grad = True
        
    for param in srgan_generator.net.subpixel_convolutional_blocks.parameters():
        param.requires_grad = True

    model = srgan_generator
    model = model.to(device)

# --- Optimizer --- #
    optimizer = Adam(model.parameters(), lr=1.e-4)

# # --- Scheduler --- #
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

# # --- Loss --- #
    loss = nn.MSELoss()

# #-------------------------------------------------------------------------------#
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
    save_steps=1
)

trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset,           # evaluation dataset
    optimizer=optimizer,
    criterion=loss,
    truncated_vgg19=None,
    scheduler=scheduler,
    epoch=epoch
)

trainer.train()
