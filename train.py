import logging
import argparse
import copy
from tqdm import tqdm
import random
from collections import namedtuple
from pathlib import Path
import wandb

import torch
import torch.nn as nn
from torchvision import transforms

from model import *

import aiutils
import aiutils.schedule as schedule
import aiutils.logger as log
from aiutils import *
from aiutils.dataset import ImageDataloader
from aiutils.losses import MSELoss
from aiutils.metrics import psnr, ms_ssim
from aiutils.transforms import CropCityscapesArtefacts

logging.getLogger().setLevel(logging.INFO)
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_defaults = {
    'project': 'cityscapes-balle',
    'entity': 'aistream',

    'train': 'cityscapes',
    'eval': 'cityscapes',
    'batch_size': 1,
    'lr': 1e-4,
    'steps': 1500000,
    'eval_steps': 1,

    'lambda_loss': 0.05,
    'debug': False,
    'device': 3,  # get_free_gpu(),
    'num_workers': 2,

    'optim': {
        'name': 'Adam',
        'kwargs': {}
    },

    'resume': False,
    'scheduler': {
        'name': 'ExponentialLR',
        'steps': 300000,
        'kwargs': {
            'gamma': 0.5,
        }
    },
    'log': {  # Log: log_steps pairs
        # SCALARS:
        'loss': 100,
        'rate': 100,
        'distortion': 100,
        'psnr': 100,
        'ms_ssim': 100,
        'bpp_z': 100,
        'bpp_feature': 100,

        # IMAGES
        'prediction': 1000
        # 'prediction_all': 1000
    },
    'eval_metric': 'loss'  # eval models are compared with respect to this metric

}


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def rate_distortion_loss(rate, distortion, l):
    return torch.mean((rate + l*distortion)/(1 + l))


class Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        #### DATALOADER ####
        data_transforms = {
            'train': transforms.Compose([
                CropCityscapesArtefacts(),
                transforms.RandomCrop(256),
                transforms.ToTensor()]),
            'eval': transforms.Compose([
                CropCityscapesArtefacts(),
                # transforms.CenterCrop(512),
                transforms.ToTensor()])
        }
        shuffle_dict = {'train': True, 'eval': False}

        self.dataloaders = {
            ds_type: ImageDataloader(ds_name=getattr(config, ds_type), ds_type=ds_type, shuffle=shuffle_dict[ds_type],
                                     transforms=data_transforms[ds_type], **config._asdict())
            for ds_type in ['train', 'eval']}

        #### MODEL, OPTIMIZER, SCHEDULER ####
        self.model = ImageCompressor(128, 192).to(self.device)
        print(
            f'Model {type(self.model).__name__} has {get_parameter_count(self.model)} trainable parameters.')
        self.optimizer = getattr(torch.optim, config.optim['name'])(
            params=self.model.parameters(), lr=config.lr, **config.optim['kwargs'])
        if config.scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = getattr(torch.optim.lr_scheduler, config.scheduler['name'])(
                optimizer=self.optimizer, **config.scheduler['kwargs'])
            schedule.every(config.scheduler['steps']).steps.do(
                self.scheduler.step)

        if config.resume:
            self._resume_checkpoint(Path(config.resume))

        # DEFINE CRITERION
        self.criterion = MSELoss().to(self.device)

        #### LOGGING ####
        if config.resume:
            self.wandb_id = self._resume_checkpoint(Path(config.resume))

        log.debug = config.debug
        if not config.debug:
            wandb.init(project=config.project, entity=config.entity,
                       config=config._asdict(), id=self.wandb_id, resume='allow')
            wandb.watch(self.model, log='all', log_freq=1000)
            wandb.save(__file__)
            self._create_output_folder(wandb.run.name, __file__)

        # schedule.every(10).steps.do(log.value_print, 'train', 'loss')
        assert config.eval_metric in config.log, f'eval metric {config.eval_metric} not logged!'
        for name in config.log:
            log_steps = config.log[name]
            schedule.every(log_steps).steps.do(log.commit_wb, 'train', name)

    def _train_epoch(self):
        self.model.train()
        logger = log.get_logger('train')

        for inputs, idx in tqdm(self.dataloaders['train']):
            inputs = inputs.to(self.device)

            # forward
            self.optimizer.zero_grad()

            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = self.model(
                inputs)
            rate = bpp
            distortion = mse_loss*255**2

            # backward
            loss = rate_distortion_loss(rate, distortion, config.lambda_loss)
            loss.backward()
            clip_gradient(self.optimizer, 5)
            self.optimizer.step()

            logger.log(loss=log.mean(loss),
                       rate=log.mean(rate),
                       distortion=log.mean(distortion),
                       bpp_z=log.mean(bpp_z),
                       bpp_feature=log.mean(bpp_feature),
                       psnr=log.mean(psnr(clipped_recon_image, inputs)),
                       ms_ssim=log.mean(ms_ssim(clipped_recon_image, inputs)),
                       prediction=log.image(
                           torch.cat((inputs, clipped_recon_image)).unsqueeze(0), nrow=2),
                       prediction_all=log.image_grid(clipped_recon_image))

            schedule.step()

    def _eval_epoch(self):
        self.model.eval()
        logger = log.get_logger('eval')

        for inputs, idx in tqdm(self.dataloaders['eval']):
            inputs = inputs.to(self.device)

            # forward
            with torch.set_grad_enabled(False):

                clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = self.model(
                    inputs)
                rate = bpp
                distortion = mse_loss*255**2
                loss = rate_distortion_loss(
                    rate, distortion, config.lambda_loss)

                logger.log(loss=log.mean(loss),
                           rate=log.mean(rate),
                           distortion=log.mean(distortion),
                           bpp_z=log.mean(bpp_z),
                           bpp_feature=log.mean(bpp_feature),
                           psnr=log.mean(
                               psnr(clipped_recon_image, inputs)),
                           ms_ssim=log.mean(
                               ms_ssim(clipped_recon_image, inputs)),
                           prediction=log.image(
                               torch.cat((inputs, clipped_recon_image)).unsqueeze(0), nrow=2),
                           prediction_all=log.image_grid(clipped_recon_image))

    def train(self):
        best_model = None
        best_result = float('inf')

        eval_scheduler = schedule.Scheduler()
        # EVALUATION METHOD

        def eval_model():
            nonlocal best_model, best_result
            self._eval_epoch()

            eval_result = log.get_value('eval', config.eval_metric)
            if eval_result < best_result:
                best_result = eval_result
                best_model = copy.deepcopy(self.model.state_dict())

                logging.info(
                    f'## New best loss = {best_result}. Save model as new best!')
                self._save_model('model_best.pt')

            for name in config.log:
                log.commit_wb('eval', name)
        eval_scheduler.every(config.eval_steps).steps.do(eval_model)

        # TRAINING LOOP
        end_epoch = (config.steps //
                     len(self.dataloaders['train'].dataset)) + 1
        for epoch in range(self.start_epoch, end_epoch):
            logging.info(f'## Epoch {epoch} / {end_epoch}')
            self._train_epoch()
            self._save_checkpoint(epoch)
            eval_scheduler.step()

            if schedule.steps > config.steps:
                break

        logging.info(f'## Training completed! best loss = {best_result}')

        return best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS)  # filter none items
    parser.add_argument('--project', type=str, help='project name')
    parser.add_argument('--entity', type=str, help='entity name')
    parser.add_argument('--train', type=str, help='Train dataset')
    parser.add_argument('--eval', type=str, help='Validation dataset')
    parser.add_argument('--device', type=int, help='The gpu to use')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--debug', action='store_true', help='Debug')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()

    # make named tuple with defaults
    config = namedtuple(
        'Config', config_defaults, defaults=config_defaults.values())(**vars(args))

    print('\n' + 15*'#' + ' Train config ' + 15*'#')
    for k in config._fields:
        print(f'{k:15}: {getattr(config, k)}')
    print(44*'#' + '\n')

    trainer = Trainer(config)
    trainer.train()
