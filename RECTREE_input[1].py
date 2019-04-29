import numpy as np
import os
import pickle
import random
from sklearn.model_selection import train_test_split
import string
import torch
import re
from torch.utils.data import Dataset, DataLoader

dataset_dir = r'train_data'

# (num_data, maxlen_of_tree, len_of_line )
class RECTREEdataset(Dataset):
    def __init__(self, dataset_dir=dataset_dir, maxlen_of_tree=300):
        self.dataset_dir = dataset_dir
        self.maxlen_of_tree = maxlen_of_tree
        self.label_dicts = self.read_labels_into_dict()
        self.file_list = self.list_file()
        self.num_data = len(self.file_list)
        self.maxlen_of_tree = maxlen_of_tree
        self.len_of_line = 10
        # self.data_list = self.load_data()


    def __getitem__(self, index):
        # 重载函数，返回值：
        # datum np.array shape = (maxlen_of_tree, 10)
        # label = np.array shape = (1,)
        # tree_len int 样本的实际长度，即除去填充的0的部分
        filename = self.dataset_dir + '\\' + self.file_list[index]
        with open(filename, 'r') as f:
            lines = f.read().strip().split('\n')
            lines = lines[4:]               #去除注释行

        for i in range(len(lines)):
            lines[i] = re.split(r'[\[\],\s]', lines[i])
            while '' in lines[i]:
                lines[i].remove('')
        datum = np.zeros([self.maxlen_of_tree, self.len_of_line])
        tree_len = len(lines) + 1                      # +1 避免0，其实不应该
        for i, item in enumerate(lines):
            datum[i] = np.array(item)
        label = np.array(self.label_dicts[int(self.file_list[index].split('.')[0])])
        return datum, label, tree_len




    def __len__(self):
        return len(self.file_list)

    def getitem(self):
        print(self.__getitem__(45670))


    def list_file(self):
        # 返回swc文件列表，其中不包括树长度大于maxlen的文件，也除去了找不到标签的文件
        file_list = [swc for swc in os.listdir(self.dataset_dir)]
        file_list_washed = []
        for swc in file_list:
            filename = self.dataset_dir + '\\' + swc
            with open(filename, 'r') as f:
                lines = f.read().strip().split('\n')
            if len(lines) <= self.maxlen_of_tree + 3:
                try:
                    self.label_dicts[int(swc.split('.')[0])]
                    file_list_washed.append(swc)
                except KeyError:
                    continue
        print(len(file_list_washed))
        return file_list_washed



    def read_labels_into_dict(self):
        # 将label读入形如{id:label}的字典
        with open('labels.txt', 'rb') as f:
            labels = pickle.load(f)
        label_dict = {'interneuron': 0, 'principal cell': 1, 'Glia': 2, 'Not reported': 3, 'sensory receptor': 4}
        labels = [[item[0], label_dict[item[1]]] for item in labels]
        hist = [0, 0, 0, 0, 0]
        for item in labels:
            hist[item[1]] += 1
        print('各类样本分布', hist)
        return dict(labels)




if __name__ == '__main__':
    c = RECTREEdataset()

