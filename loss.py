import torch
import torch.nn as nn
from utils import get_rank
import math
import numpy as np


def obj_loss(inputs, labels):
    # outputs already normalized
    criterion = nn.NLLLoss()
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    return loss


def ce_loss(inputs, labels):
    # outputs already normalized
    criterion = nn.NLLLoss()
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    return loss


def v_loss(inputs):
    device = torch.device('cuda:{}'.format(get_rank()))
    criterion = nn.MSELoss(reduce=False)
    var = torch.var(inputs, dim=1)
    var_head = torch.ones_like(var)
    var_head = var_head.to(device)
    loss = criterion(var, var_head)
    loss = loss.sum() / loss.shape[0]
    return loss


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction='none')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    loss = loss.sum() / loss.shape[0]
    return loss


def L1_loss(inputs, labels):
    criterion = nn.L1Loss(reduction='mean')
    loss = criterion(inputs, labels.float())
    return loss


def normal_sampling(mean, label_k, std=2):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)
