from torch.nn.functional import softmax
from tqdm import tqdm
import argparse
from dataset import make_dataloader
import os
import numpy as np
import torch
from utils import calculate_score, m_evaluate
from prettytable import PrettyTable
from PIL import Image
from config import cfg
from model import build_model

cfg.merge_from_file("configs/mfa.yml")
print('==========> start test model')

model = build_model(cfg)
param_dict = torch.load(cfg.TEST.WEIGHT, map_location=lambda storage, loc: storage)
print(param_dict['state_dict'])
param_dict1 = {}
for k, v in param_dict.items():
    k_ = k.replace("module.", "")
    param_dict1[k_]=param_dict[k]
print('ignore_param:')
print([k for k, v in param_dict1.items() if k not in model.state_dict() or
        model.state_dict()[k].size() != v.size()])

