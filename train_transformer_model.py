import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from poutyne import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from poutyne.framework import Model
from poutyne.framework.callbacks import TensorBoardLogger, ModelCheckpoint
# from poutyne.framework.metrics import EpochMetric
import RNNformer
from Utils import *
import Data.data as data
from Model import *
from Conform import Conformer
from Conform_AP import Conformer_AP
import math
from FEA import FEA_T
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class TopKAccuracy(nn.Module):
    def __init__(self, k: int):
        super(TopKAccuracy, self).__init__()
        self.k = k
        self.acc = None
        self.__name__ = f"top_{self.k}_accuracy"
        self.register_buffer("_accuracy", None)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """ Compute the metric """
        self._accuracy = accuracy(y_pred, y_true, topk=self.k)
        return float(self._accuracy.cpu().numpy().squeeze())

    def get_metric(self) -> float:
        """ Return the float version of the computed metric """
        return self._accuracy.numpy()

    def reset(self) -> None:
        """ Reset metric at end of each epoch """
        self._accuracy = None


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='Conformer9_2_CR=4,norescov,whole', help="Model to train")
parser.add_argument("--epochs", type=int, default=150, help="Epochs to train for")
parser.add_argument("--batch_size", type=int, default=2048, help="Number of samples in each batch (set lower to reduce CUDA memory used")
parser.add_argument("--split", type=float, default=0.8, help="Percentage of data in train set")
parser.add_argument("--dropout", type=float, default=0.25)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrf', type=float, default=0.1)
parser.add_argument('--pre_weights', type=str, default="Conformer9_2_CR=4,norescov,whole")
args = parser.parse_args()

# Make reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Params
N_CLASSES = 11
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SPLIT = args.split
DROPOUT = args.dropout
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models/models_"+args.model
LOG_DIR = os.path.join("logs/logs_"+ args.model)
PRE_WIGHT = f'models/models_{args.model}/{args.model}.pt'

# Load dataset
#net = FEA_T(device='cuda', dropout=0.6)
#net = RNNformer.RNNformer(data_length=128, in_chans=2,  channel_ratio=2, depth=9, base_channel=2, num_heads=4, mlp_ratio=4, qkv_bias=True, Device='cuda', drop_rate=0.6, drop_path_rate=0.5)
#net = FEA_T(device='cuda')
#dataset = data.RadioML2016()
#net = Resformer(device='cuda')
net = Conformer(Device='cuda', depth=9, trans2con=True, channel_ratio=4)
lf = lambda x:((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
scheduler = LambdaLR(lr_lambda=lf, last_epoch=-1)
net.train()
# Setup dataloaders
train_dataloader = data.DataLoader_train
val_dataloader = data.DataLoader_val
if os.path.exists(PRE_WIGHT):
    weights_dict = torch.load(PRE_WIGHT, map_location=('cuda'))
    net.load_state_dict(weights_dict)
    print("load pretrained")
# Callbacks
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
checkpoint = ModelCheckpoint(
    filename=os.path.join(MODEL_DIR, args.model + ".pt"),
    monitor="val_loss",
    save_best_only=True
)
writer = SummaryWriter(LOG_DIR)
tb_logger = TensorBoardLogger(writer)
callbacks = [checkpoint, tb_logger, scheduler]#

# Metrics
top3 = TopKAccuracy(k=1)
top5 = TopKAccuracy(k=3)
metrics = ["acc", top3, top5]
progra = [p for p in net.parameters() if p.requires_grad]
#optimizer = optim.SGD(progra ,lr=args.lr, momentum=0.9, weight_decay=0.01)
adam = optim.Adam(progra, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
# Train
model = Model(
    network=net,
    optimizer=adam,
    loss_function=nn.CrossEntropyLoss(),
    batch_metrics=metrics
)
model.to(DEVICE)
#model.train()
history = model.fit_generator(
    train_dataloader,
    val_dataloader,
    epochs=EPOCHS,
    callbacks=callbacks
)
print(history)
