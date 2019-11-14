# -- coding: utf-8 --
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
input_size = 224
# Data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
data_dir = {'train': ('../png_data/png_v1/train/interneuron', '../png_data/png_v1/train/principal cell'),
            'test': ('../png_data/png_v1/test/interneuron', '../png_data/png_v1/test/principal cell')}
class neuronSet(Dataset):


    def __init__(self, phase, data_dir=data_dir):
        super(neuronSet, self).__init__()
        self.phase = phase
        self.data_dir = data_dir[phase]
        self.file_list = self.list_file()

    def list_file(self):
        # 返回所有文件路径的列表
        # 格式为[[第一类的所有文件路径]，[第二类的所有文件路径]， ...]
        # 并考虑了样本均衡

        file_list = [[] for d in self.data_dir]

        for i in range(len(file_list)):
            self.recur_listdir(self.data_dir[i], file_list[i])

        # 样本均衡， 将所有类的样本数扩充到一致
        if self.phase == 'train':
            m = max([len(fi) for fi in file_list])
            for fi in file_list:
                if len(fi) < m:
                    extra = np.random.choice(fi, size=m - len(fi))
                    fi.extend(extra)

        print('pngset ', self.phase, [len(fi) for fi in file_list])

        return file_list

    def __getitem__(self, item):
        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c + 1]):
            c += 1
        filename = self.file_list[c][item - 1 - sum(len(fi) for fi in self.file_list[:c])]
        # neuro_id = int(filename.split('/')[-1].split('.')[0])
        # print('item:', item)
        # print('neuro_id_',neuro_id)

        datum = Image.open(filename).convert('RGB')
        label = c

        datum = data_transforms[self.phase](datum)
        return datum, label

    def __len__(self):
        return sum([len(l) for l in self.file_list])

    def recur_listdir(self, path, dir_list):
        # 将path下的所有文件放到dir_list 中
        for f in os.listdir(path):
            if os.path.isdir(path + '/' + f):
                self.recur_listdir(path + '/' + f, dir_list)
            else:
                dir_list.append(path + '/' + f)

    def getitem(self, item):
        '''
        similar to __gititem__, but return the neuro_id of sample
        used for multi-model
        :return: neuro_id
        '''
        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c + 1]):
            c += 1

        filename = self.file_list[c][item - 1 - sum(len(fi) for fi in self.file_list[:c])]
        neuro_id = int(filename.split('/')[-1].split('.')[0])
        #print ('neuro_id_',neuro_id)
        return neuro_id

if __name__ == '__main__':
    import os
    import random

    c = neuronSet('test')
    print(c.__getitem__(40))
    print(c.__len__())



