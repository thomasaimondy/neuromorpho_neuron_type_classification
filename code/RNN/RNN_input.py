import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
import string
import torch
import re
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


data_dir = {'train': ('swc_data/swc_v1/train/interneuron', 'swc_data/swc_v1/train/principal cell/microglia', 'swc_data/swc_v1/train/principal cell/pyramidal', 'swc_data/swc_v1/train/Glia', 'swc_data/swc_v1/train/Not reported', 'swc_data/swc_v1/train/sensory receptor'),
            'test': ('swc_data/swc_v1/test/interneuron', 'swc_data/swc_v1/test/principal cell/microglia', 'swc_data/swc_v1/test/principal cell/pyramidal', 'swc_data/swc_v1/test/Glia', 'swc_data/swc_v1/test/Not reported', 'swc_data/swc_v1/test/sensory receptor')}

class RNNdataset(Dataset):
    def __init__(self, phase, data_dir=data_dir, maxlen_of_tree=300, len_of_line=7):
        self.phase = phase
        self.data_dir = data_dir[phase]
        self.maxlen_of_tree = maxlen_of_tree
        self.file_list = self.list_file()
        self.len_of_line = len_of_line


    def __getitem__(self, item):
        # 重载函数，返回值：
        # datum np.array shape = (maxlen_of_tree, len_of_line)
        # label = np.array shape = (1,)
        # tree_len int 样本的实际长度，即除去填充的0的部分的长度

        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c+1]):
            c += 1

        filename = self.file_list[c][item - 1 - sum(len(fi) for fi in self.file_list[:c])]
        label = c

        lines = self.readswc(filename)
        datum = np.zeros([self.maxlen_of_tree, self.len_of_line])
        tree_len = len(lines)
        for i, item in enumerate(lines):
            datum[i] = np.array(item)

        return datum, label, tree_len


    def __len__(self):
        return sum([len(fi) for fi in self.file_list])

    def list_file(self):
        # 返回所有文件路径的列表， 其中不包括树长度大于maxlen的文件，因为文件太长可能会使训练太慢
        # 格式为[[第一类的所有文件路径]，[第二类的所有文件路径]， ...]
        # 并考虑了样本均衡

        ls = [[] for d in self.data_dir]
        file_list = [[] for d in self.data_dir]

        for i in range(len(ls)):
            self.recur_listdir(self.data_dir[i], ls[i])

        # wash 掉长度太长的
        for i, d in enumerate(ls):
            for filename in tqdm(d):
                with open(filename, 'r') as f:
                    lines = f.read().strip().split('\n')

                j = 0
                while lines[j].startswith('#'):
                    j += 1
                lines = lines[j:]  # 去除注释行

                if len(lines) <= self.maxlen_of_tree:
                    file_list[i].append(filename)

        # 样本均衡， 将所有类的样本数扩充到一致
        if self.phase == 'train':
            m = max([len(fi) for fi in file_list])
            for fi in file_list:
                if len(fi) == 0:
                    raise ValueError('some classes are empty')
                    # maxlen_of_tree太小，或者数据集本身有问题
                if len(fi) < m:
                    extra = np.random.choice(fi, size=m - len(fi))
                    fi.extend(extra)

        print('RNNdataset: ', self.phase, [len(fi) for fi in file_list])

        return file_list


    def recur_listdir(self, path, dir_list):
        # 将path下的所有文件放到dir_list 中
        for f in os.listdir(path):
            if os.path.isdir(path + '/' + f):
                self.recur_listdir(path + '/' + f, dir_list)
            else:
                dir_list.append(path + '/' + f)

    def readswc(self, filename):
        # 读取swc文件，返回形状为行数 * 每行长度（len_of_line）的列表
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')

        i = 0
        while lines[i].startswith('#'):
            i += 1
        lines = lines[i:]  # 去除注释行

        for i in range(len(lines)):
            lines[i] = re.split(r'[\[\],\s]', lines[i])
            while '' in lines[i]:
                lines[i].remove('')

        return lines





if __name__ == '__main__':

    c = RNNdataset('train')