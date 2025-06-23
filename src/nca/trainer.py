from pathlib import Path
import time
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self,
                 model: torch.nn.Module,
                 metrics: dict,
                 steps: int,
                 max_time: int,
                 fire_rate: float,
                 channel_n: int,
                 input_channels: int,
                 batch_size: int,
                 device: torch.device,
                 criterion: torch.nn,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 log_dir: str,
                 replace_by_targets=False,
                 acquire_at_steps=False,
    ):
        self.model = model
        self.metrics = metrics
        self.steps = steps
        self.max_time = max_time
        self.fire_rate = fire_rate
        self.channel_n = channel_n
        self.input_channels = input_channels
        self.batch_size=batch_size
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.replace_by_targets = replace_by_targets
        self.acquire_at_steps = np.array(acquire_at_steps)

        # logging
        # ---------------------------------------------------
        if log_dir:
            log_dir = Path(log_dir)
            if not log_dir.parent.exists():
                log_dir.parent.mkdir()
            self.log_dir = log_dir
            self.writer = SummaryWriter(log_dir=log_dir)

            self.logger = logging.getLogger()
            logging.basicConfig(
                handlers=[
                    logging.FileHandler(str(log_dir/'run.log')),
                    logging.StreamHandler()
                ],
                format='[{levelname}] {asctime}: {message}',
                style='{',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
            )
        # ---------------------------------------------------


    def pad_hidden(self, img):
        shape = list(img.shape)
        shape[-1] = self.channel_n
        padded = torch.zeros(shape, dtype=torch.float32).to(self.device)
        padded[..., 0:img.shape[-1]] = img
        return padded

    def prepare_data(self, data):
        inputs, targets, weights, times = data #weights == segmentation
        inputs = self.pad_hidden(data[0]) # pads and sends to device
        targets = targets.to(self.device).type(torch.float32)
        weights = weights.to(self.device).type(torch.float32)
        if np.any(self.acquire_at_steps):
            # if hard encoded acquire steps a provided use them
            acquire_at_steps = torch.from_numpy(self.acquire_at_steps)
            acquire_at_steps = acquire_at_steps.type(torch.int)
            # use only steps with a ground truth a hand
            times = torch.round(times).type(torch.int)
            acquire_at_steps = acquire_at_steps*(times != 0).type(torch.int)
        else:
            acquire_at_steps = self.steps/self.max_time * times
        acquire_at_steps = torch.round(acquire_at_steps)
        acquire_at_steps = acquire_at_steps.to(self.device).type(torch.int)
        return inputs, targets, weights, acquire_at_steps

    def one_batch_step(self, data, train=True):
        inputs, targets, weights, acquire_at = self.prepare_data(data)
        loss_masking = (acquire_at != 0).clone()
        self.optimizer.zero_grad()

        # use replace_by_targets only during training
        if train:
            replace_by_targets = self.replace_by_targets
        else:
            replace_by_targets = 0

        loss = 0.
        with torch.set_grad_enabled(train):
            outputs = self.model(inputs, acquire_at, replace_by_targets, targets=targets)
            # compute loss based on the "visual", i.e. non-hidden channels, of the output
            #loss = self.criterion(outputs[..., :self.input_channels], targets, weights)
            loss = self.criterion(outputs[..., :self.input_channels], targets, loss_masking)
        metrics = {}
        for name, metrc in self.metrics.items():
            metrics[name] = metrc(
                outputs[..., :self.input_channels],
                targets,
                loss_masking,
            ).item()*inputs.size(0)

        # backprop
        if train:
            loss.backward()
            self.optimizer.step()

        return loss.item()*inputs.size(0), metrics


    def train(self, train_loader, val_loader, n_epochs):
        since = time.time()

        best_loss = np.inf
        for epoch in range(n_epochs):
            # training
            self.model.train()
            # initialize loss and metrics
            train_loss = 0.
            train_metrics = {key: 0. for key in self.metrics.keys()}
            # iterate over loader
            for data in train_loader:
                loss, metrics = self.one_batch_step(data, train=True)
                train_loss += loss
                for key, value in metrics.items():
                    train_metrics[key] += value
            # normalize train loss and metrics
            train_loss = train_loss/(len(train_loader.dataset))
            for key in train_metrics.keys():
                train_metrics[key] /= len(train_loader.dataset)

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()
            self.writer.add_scalar(f'learning_rate', lr[0], epoch)

            # validation
            self.model.eval()
            # initialize loss and metrics
            val_loss = 0.
            val_metrics = {key: 0. for key in self.metrics.keys()}
            for data in val_loader:
                loss, metrics = self.one_batch_step(data, train=False)
                val_loss += loss
                for key, value in metrics.items():
                    val_metrics[key] += value
            # normalize val loss and metrics
            val_loss = val_loss/len(val_loader.dataset)
            for key in val_metrics.keys():
                val_metrics[key] /= len(val_loader.dataset)

            # logging
            self.logger.info(f'epoch {epoch}/{n_epochs-1}: train loss  {train_loss:.6f}, val loss {val_loss:.6f}')
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), str(self.log_dir/'best_model.pt'))
                self.logger.info('best validation loss: logging')

            self.writer.add_scalars(f'loss', {'train': train_loss, 'val': val_loss}, epoch)
            for key in self.metrics:
                self.writer.add_scalars(f'metrics/{key}', {'train': train_metrics[key], 'val': val_metrics[key]}, epoch)

        self.writer.flush()
        self.writer.close()

        time_elapsed = time.time() - since
        self.logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
