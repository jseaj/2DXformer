import torch
import math
import os
import time
import copy
import numpy as np
from lib.metrics import All_Metrics
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler, device, logger):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.logger = logger

        self.model.to(self.device)

        # self.train_per_epoch = len(train_loader)
        best_model_name = '{}-his{}hor{}-bestModel.pth'.format(args.dataset, args.history, args.horizon)
        self.best_path = os.path.join('./logs', best_model_name)

    def _log_out(self, info: str):
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def _val_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        start_time = time.time()

        with torch.no_grad():
            for i, (x, y) in enumerate(self.val_loader):
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)

                out = self.model(x).squeeze()
                out = self.scaler.inverse_transform(out)

                loss = self.loss(out, y, 0)
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        end_time = time.time()
        self._log_out("[Val]   epoch #{}/{}: loss is {:.4f}, time used {:.3f}\n"
                      .format(epoch + 1, self.args.epochs, val_loss, end_time - start_time))
        return val_loss

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader)
        )
        for i, (x, y) in bar:
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float)
            self.optimizer.zero_grad()

            out = self.model(x).squeeze()
            out = self.scaler.inverse_transform(out)
            loss = self.loss(out, y, 0)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        train_epoch_loss = total_loss / len(self.train_loader)
        end_time = time.time()
        self._log_out("[Train] epoch #{}/{}: loss is {:.4f}, time used {:.3f}"
                      .format(epoch + 1, self.args.epochs,
                              train_epoch_loss, end_time - start_time))

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')

        train_loss_list = []
        val_loss_list = []

        for epoch in range(self.args.epochs):
            train_epoch_loss = self._train_epoch(epoch)
            val_epoch_loss = self._val_epoch(epoch)

            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                # save best model
                best_model = copy.deepcopy(self.model.state_dict())

        self._log_out("Finish training, best val loss: {:.4f}\n".format(best_loss))
        # save weights to file
        torch.save(best_model, self.best_path)

        self.model.load_state_dict(best_model)
        self.test()

    def test(self):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device, dtype=torch.float)
                y = y.to(self.device, dtype=torch.float)

                out = self.model(x).squeeze()
                out = self.scaler.inverse_transform(out)

                y_true.append(y)
                y_pred.append(out)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)
        for t in range(y_true.shape[1]):
            mae, rmse = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...], 0, 0)
            self._log_out(
                "Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}".format(
                    t + 1, mae, rmse)
            )
        mae, rmse = All_Metrics(y_pred, y_true, 0, 0)
        self._log_out(
            "Average Horizon, MAE: {:.2f}, RMSE: {:.2f}".format(
                mae, rmse)
        )
