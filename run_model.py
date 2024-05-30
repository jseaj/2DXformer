import torch
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
import pandas as pd

from lib.dataloader import get_dataloader
import argparse
from Trainer import Trainer
from lib.logger import get_logger
from models.model import Model


def init_seed(seed):
    """
    Disable cudnn to maximize reproducibility
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def loss_fn(y_pred, y_true, mask_value=None):
    if mask_value is not None:
        mask = torch.gt(y_true, mask_value)
        y_pred = torch.masked_select(y_pred, mask)
        y_true = torch.masked_select(y_true, mask)
    return F.smooth_l1_loss(y_pred, y_true, reduction='mean', beta=1.0)


parser = argparse.ArgumentParser(description='run model')
# Dataset related
parser.add_argument('--dataset', default='SDWPF', type=str)
parser.add_argument('--node_num', default=134, type=int)
parser.add_argument('--input_dim', default=10, type=int)

# training related
parser.add_argument('--val_ratio', default=0.2, type=float)
parser.add_argument('--test_ratio', default=0.1, type=float)  # 2-4
parser.add_argument('--history', default=36, type=int)
parser.add_argument('--horizon', default=24, type=int)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--epochs', default=200, type=int)

# model related
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--n_layers', default=3, type=int)

args = parser.parse_args()

init_seed(10)  # setting the random seed
train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(
    args.dataset, args.history, args.horizon, args.batch_size,
    val_ratio=args.val_ratio, test_ratio=args.test_ratio, normalizer='std'
)

# initialize the model
model = Model(
    n_var=args.input_dim,
    history=args.history,
    horizon=args.horizon,
    d_model=args.d_model,
    n_layers=args.n_layers,
    attn_dropout=0.,
    ff_dropout=0.
)

# Set training related parameters
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5*1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[50],
    gamma=0.1
)
loss_criterion = loss_fn

# create a log instance
logger = get_logger("./logs", name=__name__)

trainer = Trainer(
    model=model,
    loss=loss_criterion,
    optimizer=optimizer,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    test_loader=test_dataloader,
    scaler=scaler,
    args=args,
    lr_scheduler=lr_scheduler,
    device=device,
    logger=logger
)
trainer.train()

