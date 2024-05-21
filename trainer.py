"""
The Trainer class, to easily train a super-image model from scratch.
The design is inspired by the HuggingFace transformers library at
https://github.com/huggingface/transformers/.
"""

import os
import copy
import logging
from typing import Optional, Union, Dict, Callable

from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import *

from super_image.modeling_utils import PreTrainedModel
from super_image.configuration_utils import PretrainedConfig
from super_image.file_utils import (
    WEIGHTS_NAME,
    WEIGHTS_NAME_SCALE,
    CONFIG_NAME
)
from super_image.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    set_seed
)
from super_image import TrainingArguments
from super_image.utils.metrics import AverageMeter, compute_metrics

from utils import LoggerCSV, convert_image

class Trainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        train_dataset: Dataset = None,
        eval_dataset: Optional[Dataset] = None,
        optimizer = None,
        criterion = None,
        scheduler = None,
        truncated_vgg19 = None,
        epoch = 0
    ):
        if args is None:
            output_dir = "tmp_trainer"
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)

        if model is None:
            raise RuntimeError("`Trainer` requires a `model`")

        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.best_epoch = 0
        self.best_metric = 0.0

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.truncated_vgg19 = truncated_vgg19
        self.epoch = epoch

        # self.logger = LoggerCSV(model.model_name)

    def train(
            self, resume_from_checkpoint: Optional[Union[str, bool]] = None,
            **kwargs,
    ):
        """
        Main training entry point.
        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~super_image.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~super_image.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        args = self.args

        epochs_trained = self.epoch
        device = args.device
        num_train_epochs = args.num_train_epochs
        learning_rate = args.learning_rate
        train_batch_size = args.train_batch_size
        train_dataset = self.train_dataset
        train_dataloader = self.get_train_dataloader()
        step_size = int(len(train_dataset) / train_batch_size * 200)

        if args.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        for epoch in range(epochs_trained, num_train_epochs):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate * (0.1 ** (epoch // int(num_train_epochs * 0.8)))

            self.model.train()
            epoch_losses = AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % train_batch_size)) as t:
                t.set_description(f'epoch: {epoch}/{num_train_epochs - 1}')

                for data in train_dataloader:
                    inputs, labels = data

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    inputs = convert_image(inputs, source='[0, 1]', target='imagenet-norm')
                    labels = convert_image(labels, source='[0, 1]', target='[0, 1]')

                    preds = self.model(inputs)
                    if self.truncated_vgg19 == None:
                        preds = convert_image(preds, source='[-1, 1]', target='[0, 1]')
                        loss = self.criterion(preds, labels)
                    else:
                        # preds = convert_image(preds, source='[0, 1]', target='imagenet-norm')
                        preds = self.truncated_vgg19(preds)
                        labels = self.truncated_vgg19(labels).detach()  # detached because they're constant, targets
                        loss = self.criterion(preds, labels)

                    epoch_losses.update(loss.item(), len(inputs))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                    t.update(len(inputs))

            self.eval(epoch)

    def eval(self, epoch):
        args = self.args

        if isinstance(self.model, nn.DataParallel):
            scale = self.model.module.config.scale
        else:
            scale = 4
        device = args.device
        eval_dataloader = self.get_eval_dataloader()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()

        self.model.eval()

        epoch_losses = AverageMeter()

        with tqdm(total=(len(self.eval_dataset) - len(self.eval_dataset) % 1)) as t:
            t.set_description(f'epoch: {epoch}/{self.args.num_train_epochs - 1}')
            for data in eval_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = convert_image(inputs, source='[0, 1]', target='imagenet-norm')
                labels = convert_image(labels, source='[0, 1]', target='[0, 1]]')

                with torch.no_grad():
                    preds = self.model(inputs)

                    if self.truncated_vgg19 == None:
                        preds = convert_image(preds, source='[-1, 1]', target='[0, 1]')
                        loss = self.criterion(preds, labels)
                    else:
                        # preds = convert_image(preds, source='[0, 1]', target='imagenet-norm')
                        preds_vgg_space = self.truncated_vgg19(preds)
                        labels_vgg_space = self.truncated_vgg19(labels).detach()  # detached because they're constant, targets
                        loss = self.criterion(preds_vgg_space, labels_vgg_space)
                
                epoch_losses.update(loss.item(), len(inputs))

                metrics = compute_metrics(EvalPrediction(predictions=preds, labels=labels), scale=scale)

                epoch_psnr.update(metrics['psnr'], len(inputs))
                epoch_ssim.update(metrics['ssim'], len(inputs))
                t.set_postfix(psnr=f'{epoch_psnr.avg:.6f}')
                t.update(len(labels))
            del inputs, labels, preds

        print(f'scale:{str(scale)}      eval psnr: {epoch_psnr.avg:.2f}     ssim: {epoch_ssim.avg:.4f}')

        if epoch_psnr.avg > self.best_metric:
            self.best_epoch = epoch
            self.best_metric = epoch_psnr.avg

            print(f'best epoch: {epoch}, psnr: {epoch_psnr.avg:.6f}, ssim: {epoch_ssim.avg:.6f}')
            self.save_model(epoch)

    def _load_state_dict_in_model(self, state_dict):
        load_result = self.model.load_state_dict(state_dict, strict=False)

    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)

    def save_model(self, epoch: int, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        Will only save from the main process.
        """

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if not isinstance(self.model, PreTrainedModel):
            # Setup scale
            scale = 4
            if scale is not None:
                weights_name = WEIGHTS_NAME_SCALE.format(scale=scale)
            else:
                weights_name = WEIGHTS_NAME

            # weights = copy.deepcopy(self.model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.criterion
                }, os.path.join(output_dir, weights_name))
        else:
            self.model.save_pretrained(output_dir)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        return DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.
        """

        eval_dataset = self.eval_dataset
        if eval_dataset is None:
            eval_dataset = self.train_dataset

        return DataLoader(
            dataset=eval_dataset,
            batch_size=1,
        )
