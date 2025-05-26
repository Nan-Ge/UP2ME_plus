import numpy as np
import torch
import json
import argparse

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {2: args.learning_rate * 0.5 ** 1, 4: args.learning_rate * 0.5 ** 2,
                     6: args.learning_rate * 0.5 ** 3, 8: args.learning_rate * 0.5 ** 4,
                     10: args.learning_rate * 0.5 ** 5}
    elif args.lradj == 'type2':
        lr_adjust = {5: args.learning_rate * 0.5 ** 1, 10: args.learning_rate * 0.5 ** 2,
                     15: args.learning_rate * 0.5 ** 3, 20: args.learning_rate * 0.5 ** 4,
                     25: args.learning_rate * 0.5 ** 5}
    else:
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.best_model = model
        self.val_loss_min = val_loss

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def string_split(str_for_split):
    str_no_space = str_for_split.replace(' ', '')
    str_split = str_no_space.split(',')
    value_list = [eval(x) for x in str_split]

    return value_list

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolen type')

class Args:
    def __init__(self, paras_json, task):

        with open(paras_json, 'r') as f:
            params = json.load(f)     

        if task == 'pretrain':
            self.data_format = params.get('data_format', 'csv')  # data format
            self.data_name = params.get('data_name', 'ETTm1')  # data name
            self.root_path = params.get('root_path', './datasets/ETT/')  # root path of the data file
            self.data_path = params.get('data_path', 'ETTm1.csv')  # data file
            self.data_split = params.get('data_split', '34560,11520,11520')  # train/val/test split
            self.valid_prop = params.get('valid_prop', 0.2)  # proportion of validation set, for numpy data only
            self.checkpoints = params.get('checkpoints', './pretrain-library/')  # location to store model checkpoints
            
            self.position = params.get('position', 'abs')  # position embedding method
            self.data_dim = params.get('data_dim', 7)  # Number of dimensions of the MTS data (D)
            self.patch_size = params.get('patch_size', 12)  # patch size
            self.min_patch_num = params.get('min_patch_num', 20)  # minimum number of patches in a sampled series
            self.max_patch_num = params.get('max_patch_num', 200)  # maximum number of patches in a sampled series
            self.mask_ratio = params.get('mask_ratio', 0.5)  # mask ratio of the patch
            
            self.d_model = params.get('d_model', 256)  # dimension of hidden states (d_model)
            self.d_ff = params.get('d_ff', 512)  # dimension of MLP in transformer
            self.n_heads = params.get('n_heads', 4)  # num of heads
            self.e_layers = params.get('e_layers', 4)  # num of encoder layers (N)
            self.d_layers = params.get('d_layers', 1)  # num of decoder layers (N)
            self.dropout = params.get('dropout', 0.0)  # dropout
            
            self.efficient_loader = params.get('efficient_loader', True)  # whether to use efficient data loader, if False, use the original implement
            self.resample_patch_num = params.get('resample_patch_num', False)  # periodically resample the number of patches in a series, for large datasets
            self.pool_size = params.get('pool_size', 10)  # size of the pool for resampling
            self.resample_freq = params.get('resample_freq', 5000)  # resampling frequency
            self.num_workers = params.get('num_workers', 0)  # data loader num workers
            
            self.batch_size = params.get('batch_size', 256)  # batch size of train input data
            self.train_steps = params.get('train_steps', 500000)  # train steps
            self.learning_rate = params.get('learning_rate', 1e-4)  # optimizer initial learning rate
            self.itr = params.get('itr', 1)  # experiments times
            self.valid_freq = params.get('valid_freq', 5000)  # validating frequency
            self.valid_sep_point = params.get('valid_sep_point', 10)  # equally sample patch nums for validation
            self.valid_batches = params.get('valid_batches', -1)  # validating batches, -1 means all
            self.tolerance = params.get('tolerance', 10)  # tolerance for early stopping
            
            self.use_gpu = params.get('use_gpu', True)  # use gpu
            self.gpu = params.get('gpu', 0)  # gpu
            self.use_multi_gpu = params.get('use_multi_gpu', False)  # use multiple gpus
            self.devices = params.get('devices', '0,1,2,3')  # device ids of multiple gpus
            self.device_ids = ""  # device ids (used when multi-gpu is enabled)
        
            self.label = params.get('label', 'Sliding-Window')  # labels to attach to setting

        else:
            print(f"Task error: {task}")
            exit(1)
            

    def __repr__(self):
        return f"Args({self.__dict__})"