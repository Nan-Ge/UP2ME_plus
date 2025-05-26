import argparse
import os
import torch
import random
import numpy as np

from exp.exp_pretrain import UP2ME_exp_pretrain
from utils.tools import string_split, str2bool, Args

# Load parameters from JSON file
args = Args(
    paras_json="./scripts/pretrain_scripts/parameters/pretrain_test.json",
    task='pretrain'
)

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

# Fix random seed
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True

# Pretrain
exp = UP2ME_exp_pretrain(args)

for ii in range(args.itr):
    setting = 'U2M{}_data{}_dim{}_patch{}_minPatch{}_maxPatch{}_mask{}_dm{}_dff{}_heads{}_eLayer{}_dLayer{}_dropout{}'.format(args.label, args.data_name, 
                args.data_dim, args.patch_size, args.min_patch_num, args.max_patch_num, args.mask_ratio,
                args.d_model, args.d_ff, args.n_heads, args.e_layers, args.d_layers, args.dropout)
    
    print('>>>>>>>start pre-training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.pre_train(setting)