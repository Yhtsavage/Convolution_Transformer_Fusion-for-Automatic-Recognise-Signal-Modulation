from tqdm import tqdm
import sys
import torch
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import math
from typing import Iterable, Optional
from timm.utils import accuracy, ModelEma


def accuracy(output, target, topk=1):       
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = topk
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:maxk].reshape(-1).float().sum(0, keepdim=True)
        correct =  correct_k.mul_(100.0 / batch_size)
        return correct


@torch.no_grad()
def evaluate(model,
             criterion, data_loader, device):
    model.eval()
    #model.to(device)
    mean_acc = torch.zeros(1).to(device)
    val_loss = torch.zeros(1).to(device)
    #cosine_simila = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, desc="testing...", file=sys.stdout)
    for iter, data in enumerate(data_loader):
        inputs = data[0].to(device)
        labels = data[1].to(device)
        pred1 = model(inputs)
        acc1 = accuracy(output=pred1, target=labels)
        # acc2 = accuracy(output=pred2, target=labels)
        # pred = torch.cat((pred1, pred2), dim=0)
        # labels = torch.cat((labels, labels), dim=0)
        loss = criterion(pred1, labels)
        val_loss = ((val_loss * iter) + loss) / (iter + 1)
        mean_acc = ((mean_acc *iter) + (acc1) / (iter + 1))
        data_loader.desc = f"acc1={acc1.item()}, val_loss={val_loss}"
    return mean_acc, val_loss


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    set_training_mode=True
                    ):
    # TODO fix this for finetuning
    model.train(set_training_mode)
    criterion.train()
    data_loader = tqdm(data_loader, file=sys.stdout)
    mean_loss = torch.zeros(1).to(device)
    mean_acc = torch.zeros(1).to(device)
    for iter, data in enumerate(data_loader):
        inputs = data[0].to(device, non_blocking=True)
        label = data[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            if isinstance(outputs, list):
                pred = torch.cat((outputs[0], outputs[1]), dim=0)
                labels = torch.cat((label, label), dim=0)
                loss = criterion(pred, labels)
                acc1 = accuracy(output=outputs[0], target=label)
                acc2 = accuracy(output=outputs[1], target=label)
                batch_acc = (acc1 + acc2) / 2
            else:
                loss = criterion(outputs, label)
                batch_acc = accuracy(outputs, label)

        loss_value = loss.item()
        mean_loss = (mean_loss * iter + loss_value) / (iter + 1)  # 前+现在/经历的batch数
        mean_acc = round((((mean_acc * iter + batch_acc) / (iter + 1))).item(), 3)
        data_loader.desc = f"[epoch{epoch}] mean_loss{round(mean_loss.item(), 4)}, mean_acc={mean_acc}"
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order) #loss_G.backward() ；optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

    # gather the stats from all processes
    return mean_loss, mean_acc

if __name__ == '__main__':
    a = torch.randn((4, 10))
    b = torch.tensor([[3],
                      [4],
                      [5],
                      [6]])
    print(accuracy(a, b, 1))
    print(torch.__version__)