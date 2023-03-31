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
#model = build_model(cfg)
    # summary(model.cuda(), (3, 1024, 2048))
print('load from: ', cfg.TEST.WEIGHT)
model = torch.load(cfg.TEST.WEIGHT, map_location=lambda storage, loc: storage)
print(model.pa)

