from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4, modal=1):
        self.data_source = data_source
        self.pid_index = defaultdict(list)
        #self.cam_index = defaultdict(list)
        self.num_instances = num_instances
        self.modal = modal

        for index, (fname, pid, cam, i, accum_y) in enumerate(data_source):
            if (pid<0): continue
            if (cam+1 != self.modal): continue
            
            self.pid_index[pid].append(index)
            

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist() #将0~n-1(包括0和n-1)随机打乱后获得的数字序列
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]]) #从随机选择的身份self.pids[kid]中随机选择一个index i

            _, i_pid, _, _, _= self.data_source[i]

            ret.append(i)

            #pid_i = self.index_pid[i] #得到index i的pid
            index = self.pid_index[i_pid] #得到pid中的index列表

            select_indexes = No_index(index, i)
            if (not select_indexes): continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)
            
            for kk in ind_indexes:
                ret.append(index[kk])


        return iter(ret)
