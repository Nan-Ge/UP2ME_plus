import os
import numpy as np
import pandas as pd
import json
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

#<---------------------original dataset(loader) to get the main result in the paper, generate one seprate dataset for each window length----------------------->

#for datasets saved in csv format, mostly datasets originally used for forecasting
class Pretrain_Dataset_csv(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', ts_len=12 * 10,
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        self.ts_len = ts_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        #int split, e.g. [34560,11520,11520] for ETTm1
        if (self.data_split[0] > 1):
            train_ts_len = self.data_split[0]
            val_ts_len = self.data_split[1]
            test_ts_len = self.data_split[2]
        #ratio split, e.g. [0.7, 0.1, 0.2] for Weather
        else:
            train_ts_len = int(len(df_raw) * self.data_split[0])
            test_ts_len = int(len(df_raw) * self.data_split[2])
            val_ts_len = len(df_raw) - train_ts_len - test_ts_len

        border1s = [0, train_ts_len, train_ts_len + val_ts_len]
        border2s = [train_ts_len, train_ts_len + val_ts_len, train_ts_len + val_ts_len + test_ts_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]  #leave the first column (Timestamp)
        self.data_dim = len(cols_data)

        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]  #use training set for standardlization
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.window_num = len(self.data_x) - self.ts_len + 1

    def __getitem__(self, index):
        #get one decoupled univariate series and the corresponding channel index
        channel_idx = index // self.window_num
        window_idx = index % self.window_num

        s_begin = window_idx
        s_end = s_begin + self.ts_len

        ts_x = self.data_x[s_begin:s_end, channel_idx]

        return ts_x, channel_idx

    def __len__(self):
        return self.data_dim * self.window_num


#for datasets saved in npy format, mostly datasets originally used for anomaly detection
class Pretrain_Dataset_npy(Dataset):
    '''
    For dataset stored in .npy files, which is usually saved in '*_train.npy', '*_test.npy' and '*_test_label.npy'
    We split the original training set into training and validation set
    '''
    def __init__(self, root_path, data_name='SMD', flag="train", ts_len=10 * 5,
                 valid_prop=0.2, scale=True, scale_statistic=None):
        self.flag = flag
        self.ts_len = ts_len
        self.valid_prop = valid_prop
        self.scale = scale
        self.scale_statistic = scale_statistic
        if flag == 'train' or flag == 'val':
            data_file = os.path.join(root_path, '{}_train.npy'.format(data_name))
            label_file = None
        elif flag == 'test':
            data_file = os.path.join(root_path, '{}_test.npy'.format(data_name))
            label_file = os.path.join(root_path, '{}_test_label.npy'.format(data_name))
        self.__read_data__(data_file, label_file)

    def __read_data__(self, data_file, label_file=None):
        raw_data = np.load(data_file)

        if (self.flag == 'train' or self.flag == 'val'):
            data_len = len(raw_data)
            train_data = raw_data[0:int(data_len * (1 - self.valid_prop))]
            val_data = raw_data[int(data_len * (1 - self.valid_prop)):]
            self.train = train_data
            self.val = val_data
        elif (self.flag == 'test'):
            self.test = raw_data
            self.test_labels = np.load(label_file)

        self.data_dim = raw_data.shape[1]

        if self.scale:
            if self.flag == 'train' or self.flag == 'val':
                self.scaler = StandardScaler()
                self.scaler.fit(self.train)
                self.train = self.scaler.transform(self.train)
                self.val = self.scaler.transform(self.val)

            elif self.flag == 'test':
                # use pre-computed mean and std
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
                self.test = self.scaler.transform(self.test)

        if self.flag == 'train':
            self.window_num = len(self.train) - self.ts_len + 1
            self.data_x = self.train
        elif self.flag == 'val':
            self.window_num = len(self.val) - self.ts_len + 1
            self.data_x = self.val
        elif self.flag == 'test':
            self.window_num = len(self.test) - self.ts_len + 1
            self.data_x = self.test

    def __len__(self):
        return self.data_dim * self.window_num

    def __getitem__(self, index):
        channel_idx = index // self.window_num
        window_idx = index % self.window_num

        s_begin = window_idx
        s_end = s_begin + self.ts_len

        ts_x = self.data_x[s_begin:s_end, channel_idx]

        return ts_x, channel_idx

#<---------------------efficient dataset(loader) maintain only one data copy and sample the desired window length on fly----------------------->
class Pretrain_Dataset_csv_efficient(Dataset):
    '''
    An efficient dataloader, instead of maintaining one dataloder for each window length, we maintain only one copy of data and samples the desired length
    '''
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train',
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.scale_statistic = scale_statistic

        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split

        self.__read_data__()

    def __read_data__(self):
        self.dataset = None
        self.data_dim = None
        
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        data_split = self.data_split
        
        #int split, e.g. [34560,11520,11520] for ETTm1
        if (data_split[0] > 1):
            train_ts_len = data_split[0]
            val_ts_len = data_split[1]
            test_ts_len = data_split[2]
        #ratio split, e.g. [0.7, 0.1, 0.2] for Weather
        else:
            train_ts_len = int(len(df_raw) * data_split[0])
            test_ts_len = int(len(df_raw) * data_split[2])
            val_ts_len = len(df_raw) - train_ts_len - test_ts_len

        border1s = [0, train_ts_len, train_ts_len + val_ts_len]
        border2s = [train_ts_len, train_ts_len + val_ts_len, train_ts_len + val_ts_len + test_ts_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        self.data_dim = len(cols_data)
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        print('data points:', len(data[border1:border2]))
        self.dataset = data[border1:border2]
    
    def get_data_dim(self):
        return self.data_dim
    
    def get_start_limit(self, ts_len):
        return len(self.dataset) - ts_len + 1

    def random_sample_index(self, batch_size, ts_len):
        #sample a batch of time series with length ts_len from the dataset

        #generate start index and channel index
        start_limit = self.get_start_limit(ts_len)
        start_idxs = np.random.choice(start_limit, batch_size)
        data_dim = self.get_data_dim()
        channel_idxs = np.random.choice(data_dim, batch_size)
        
        return start_idxs, channel_idxs
    
    def get_batch(self, ts_len, start_idxs, channel_idxs):
        #get a batch of time series with the sampled indexs

        batched_ts = []
        for i in range(len(start_idxs)):
            batched_ts.append(torch.from_numpy(self.dataset[start_idxs[i]:start_idxs[i]+ts_len, channel_idxs[i]]))
        batched_ts = torch.stack(batched_ts)
        batched_channel_idxs = torch.from_numpy(channel_idxs)

        return batched_ts, batched_channel_idxs
    
    def get_random_batch(self, batch_size, ts_len):
        start_idxs, channel_idxs = self.random_sample_index(batch_size, ts_len)
        batched_ts, batched_channel_idxs = self.get_batch(ts_len, start_idxs, channel_idxs)

        return batched_ts, batched_channel_idxs
    
    def set_iteration_params(self, ts_len, shuffle=False):
        #go through the dataset with ts_len, for validation and testing
        
        start_limit = self.get_start_limit(ts_len)
        data_dim = self.get_data_dim()
        #all possible combinations: T * N
        self.candidate_idxs = np.zeros((start_limit * data_dim, 2), dtype=int)
        for d in range(data_dim):
            self.candidate_idxs[d * start_limit:(d + 1) * start_limit, 0] = np.arange(start_limit)
            self.candidate_idxs[d * start_limit:(d + 1) * start_limit, 1] = d
        if shuffle:
            np.random.shuffle(self.candidate_idxs)

        self.ts_len = ts_len

        self.pointer = 0
        
        return 
    
    def get_next_batch(self, batch_size):
        finised = False
        start_point = self.pointer
        end_point = min(start_point + batch_size, len(self.candidate_idxs))
        if end_point == len(self.candidate_idxs):
            finised = True
        
        batch_idxs = self.candidate_idxs[start_point:end_point]
        batched_ts, batched_channel_idxs = self.get_batch(self.ts_len, batch_idxs[:, 0], batch_idxs[:, 1])

        self.pointer = end_point

        return batched_ts, batched_channel_idxs, finised
    

class Pretrain_Dataset_npy_efficient(Dataset):
    '''
    An efficient dataloader, instead of maintaining one dataloder for each window length, we maintain only one copy of data and samples the desired length

    '''
    def __init__(self, root_path, data_name='SMD', flag="train",
                 valid_prop=0.2, scale=True, scale_statistic=None):
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.valid_prop = valid_prop
        self.scale = scale
        self.scale_statistic = scale_statistic
        self.__read_data__(root_path, data_name)

    def __read_data__(self, root_path, data_name):
        self.dataset = None
        self.data_dim = None

        train_data_file = os.path.join(root_path, '{}_train.npy'.format(data_name))
        test_data_file = os.path.join(root_path, '{}_test.npy'.format(data_name))

        raw_train_val_data = np.load(train_data_file)
        raw_test_data = np.load(test_data_file)
        train_val_len = len(raw_train_val_data)
        raw_train_data = raw_train_val_data[0:int(train_val_len * (1 - self.valid_prop))]
        raw_val_data = raw_train_val_data[int(train_val_len * (1 - self.valid_prop)):]
        data_map = {'train': raw_train_data, 'val': raw_val_data, 'test': raw_test_data}
        
        self.data_dim = raw_train_val_data.shape[1]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                self.scaler.fit(raw_train_data)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            data = self.scaler.transform(data_map[self.flag])
        else:
            data = data_map[self.flag]
            
        print('data points:', len(data))
        self.dataset = data
    
    def get_data_dim(self):
        return self.data_dim
    
    def get_start_limit(self, ts_len):
        return len(self.dataset) - ts_len + 1

    def random_sample_index(self, batch_size, ts_len):
        #sample a batch of time series with length ts_len from the dataset

        #generate start index and channel index
        start_limit = self.get_start_limit(ts_len)
        start_idxs = np.random.choice(start_limit, batch_size)
        data_dim = self.get_data_dim()
        channel_idxs = np.random.choice(data_dim, batch_size)
        
        return start_idxs, channel_idxs
    
    def get_batch(self, ts_len, start_idxs, channel_idxs):
        #get a batch of time series with the sampled indexs

        batched_ts = []
        for i in range(len(start_idxs)):
            batched_ts.append(torch.from_numpy(self.dataset[start_idxs[i]:start_idxs[i]+ts_len, channel_idxs[i]]))
        batched_ts = torch.stack(batched_ts)
        batched_channel_idxs = torch.from_numpy(channel_idxs)

        return batched_ts, batched_channel_idxs
    
    def get_random_batch(self, batch_size, ts_len):
        start_idxs, channel_idxs = self.random_sample_index(batch_size, ts_len)
        batched_ts, batched_channel_idxs = self.get_batch(ts_len, start_idxs, channel_idxs)

        return batched_ts, batched_channel_idxs
    
    def set_iteration_params(self, ts_len, shuffle=False):
        #go through the dataset with ts_len, for validation and testing
        
        start_limit = self.get_start_limit(ts_len)
        data_dim = self.get_data_dim()
        #all possible combinations: T * N
        self.candidate_idxs = np.zeros((start_limit * data_dim, 2), dtype=int)
        for d in range(data_dim):
            self.candidate_idxs[d * start_limit:(d + 1) * start_limit, 0] = np.arange(start_limit)
            self.candidate_idxs[d * start_limit:(d + 1) * start_limit, 1] = d
        if shuffle:
            np.random.shuffle(self.candidate_idxs)

        self.ts_len = ts_len

        self.pointer = 0
        
        return 
    
    def get_next_batch(self, batch_size):
        finised = False
        start_point = self.pointer
        end_point = min(start_point + batch_size, len(self.candidate_idxs))
        if end_point == len(self.candidate_idxs):
            finised = True
        
        batch_idxs = self.candidate_idxs[start_point:end_point]
        batched_ts, batched_channel_idxs = self.get_batch(self.ts_len, batch_idxs[:, 0], batch_idxs[:, 1])

        self.pointer = end_point

        return batched_ts, batched_channel_idxs, finised