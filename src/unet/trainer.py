from pathlib import Path
import time
import logging

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(
        self,
        model: torch.nn.Module,
        metrics: dict,
        batch_size: int,
        device: torch.device,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        log_dir: str):

        self.model = model
        self.metrics = metrics
        self.batch_size = batch_size
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_dir = log_dir

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


    def prepare_data(self, data):
        inputs, targets, weights, times = data

        inputs = inputs.to(self.device).type(torch.float32)
        targets = targets.to(self.device).type(torch.float32)
        weights = weights.to(self.device).type(torch.float32)
        times = times.to(self.device).type(torch.float32)

        return inputs, targets, weights, times

    def one_batch_step(self, data, train=True):
        inputs, targets, weights, times = self.prepare_data(data)
        loss_masking = (times != 0).clone()
        self.optimizer.zero_grad()

        loss = 0.
        with torch.set_grad_enabled(train):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets, loss_masking)

        # backprop
        if train:
            loss.backward()
            self.optimizer.step()

        metrics = {}
        for name, metrc in self.metrics.items():
            metrics[name] = metrc(
                outputs,
                targets,
                loss_masking,
            ).item()*inputs.size(0)

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



#    def train(self, train_loader, val_loader, n_epochs):
#        since = time.time()
#
#        best_loss = np.inf
#        for epoch in range(n_epochs):
#            # training
#            self.model.train()
#            train_loss = 0.
#            for data in train_loader:
#                train_loss += self.one_batch_step(data, train=True)
#            train_loss = train_loss/(len(train_loader))
#
#            self.scheduler.step()
#            lr = self.scheduler.get_last_lr()
#            self.writer.add_scalar(f'learning_rate', lr[0], epoch)
#            self.logger.info(f'Epoch {epoch}/{n_epochs-1}: Train Loss {train_loss:.6f}')
#
#            # validation
#            self.model.eval()
#            val_loss = 0.
#            for data in val_loader:
#                val_loss += self.one_batch_step(data, train=False)
#            val_loss = val_loss/len(val_loader)
#
#            # logging
#            self.logger.info(f'Epoch {epoch}/{n_epochs-1}: Val Loss {val_loss:.6f}')
#            if val_loss < best_loss:
#                best_loss = val_loss
#                torch.save(self.model.state_dict(), str(self.log_dir/'best_model.pt'))
#                self.logger.info('best validation loss --> logging')
#
#            self.writer.add_scalars(f'loss', {'train': train_loss, 'val': val_loss}, epoch)
#
#            #log last epoch
#            if epoch == (n_epochs - 1):
#                torch.save(self.model.state_dict(), str(self.log_dir/'last_epoch.pt'))
#
#
#        self.writer.flush()
#        self.writer.close()
#
#        time_elapsed = time.time() - since
#        self.logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
