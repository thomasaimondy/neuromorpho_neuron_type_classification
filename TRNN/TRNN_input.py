import numpy as np
import os
import pickle
from tqdm import tqdm
from TRNN.TRNN_model import Tree
import torch
import re
import random
from torch.utils.data import Dataset, DataLoader


data_dir = {'train': ('../swc_data/swc_v0/train/interneuron', '../swc_data/swc_v0/train/principal cell'),
            'test': ('../swc_data/swc_v0/test/interneuron', '../swc_data/swc_v0/test/principal cell')}


class TRNNdataset(Dataset):
    def __init__(self, phase, data_dir=data_dir, maxlen_of_tree=300, len_of_line=7):
        self.phase = phase
        self.data_dir = data_dir[phase]
        self.maxlen_of_tree = maxlen_of_tree
        self.file_list = self.list_file()
        self.len_of_line = len_of_line

    def __getitem__(self, item):
        # 重载函数，返回值：
        # datum np.array shape = (maxlen_of_tree, 7)
        # label = np.array shape = (1,)
        # tree_len int 样本的实际长度，即除去填充的0的部分

        c = 0
        while item > sum(len(fi) for fi in self.file_list[0:c + 1]):
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

        # 返回swc文件列表，其中不包括树长度大于maxlen的文件

        print('reading data list ... ')
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
                if len(fi) < m:
                    extra = np.random.choice(fi, size=m - len(fi))
                    fi.extend(extra)

        print('TRNNdataset: ', self.phase, [len(fi) for fi in file_list])

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

    def lines2tree(self, lines):

        # 寻找根节点
        i2p = dict([[p[0], p[6]] for p in lines])  # 根据行数i返回其parent的行数p
        p2t = dict()  # 根据行数p返回其对应的树t
        global root
        for i in i2p:
            if int(i2p[i]) == -1:
                root = Tree(lines[int(i - 1)])
                p2t[i] = root
                break

        def add_node(i, line):
            temp = Tree(line)
            if int(i2p[i]) not in p2t:  # 如果待入树节点的父亲还没有加入树，递归加入
                add_node(int(i2p[i]), lines[int(i2p[i])])
            p2t[int(i2p[i])].add_child(temp)
            p2t[i] = temp

        for i, line in enumerate(lines):
            # print(i, p2t)
            if int(i2p[i + 1]) == -1:  # 如果是根节点， 跳过
                continue
            if i + 1 not in p2t:  # 如果第i行还没有成为树节点
                add_node(i + 1, line)  # 将它父亲所在的行生成节点并加入树

        return root

    def index_error(self, lines):
        # 检查lines所表示的是否是一棵树（是否有环）
        i = 0
        while lines[i].startswith('#'):
            i += 1
        lines = lines[i:]  # 去除注释行

        for i in range(len(lines)):
            lines[i] = re.split(r'[\[\],\s]', lines[i])
            while '' in lines[i]:
                lines[i].remove('')
        lines = np.array(lines, dtype='double')
        try:
            tree = self.lines2tree(lines)
        except RecursionError as e:
            print(e)
            return True
        except IndexError as e:
            print(e)
            return True
        return False

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
        return neuro_id





if __name__ == '__main__':
    # 改用lstm,  没有明显效果
    # 归一化，
    # 去掉某些项，比如id   去掉id和坐标后几乎无区别 / 只保留坐标情况下 似乎训练慢一些， 区别不大 / 全部为1或random， 达不到60
    # 比较两棵树相似度的方法， 把树转化为字符串

    # 数据可视化
    # 应该查看tree rnn的梯度传播，考虑树太大时候的0梯度问题

    # 生成一些假的神经元，生成对抗
    # 先训练一个判别模型，
    # 然后做一个生成模型

    from mpl_toolkits import mplot3d #有用
    import matplotlib.pyplot as plt

    c = TRNNdataset('test', renew=True)
    print(c.__getitem__(3))

    # 可视化， 观察数据长度
    # len0 = []
    # len1 = []
    # for file in c.file_list:
    #     filename = c.dataset_dir + '/' + file
    #     with open(filename, 'r') as f:
    #         lines = f.read().strip().split('\n')
    #         lines = lines[4:]  # 去除注释行
    #
    #     for i in range(len(lines)):
    #         lines[i] = re.split(r'[\[\],\s]', lines[i])
    #         while '' in lines[i]:
    #             lines[i].remove('')
    #     tree_len = len(lines)
    #     label = c.label_dicts[int(file.split('.')[0])]
    #     if label == 0:
    #         len0.append(tree_len)
    #     elif label == 1:
    #         len1.append(tree_len)
    #     else:
    #         print(label)
    # plt.hist(x=[len0, len1], bins=50)
    # plt.show()



    # 可视化 ###########################################################

    # ax = plt.axes(projection='3d')
    #
    # filename = dataset_dir + '/' + '100.swc'
    # with open(filename, 'r') as f:
    #     lines = f.read().strip().split('\n')
    #     lines = lines[4:]               #去除注释行
    #
    # for i in range(len(lines)):
    #     lines[i] = re.split(r'[\[\],\s]', lines[i])
    #     while '' in lines[i]:
    #         lines[i].remove('')
    # print(lines)
    # lines = np.array(lines, dtype='float32')
    # print(lines)
    # from sklearn.preprocessing import normalize
    #
    # # ax.plot3D(lines[:, 2], lines[:, 3], lines[:, 4], 'gray')
    # ax.scatter3D(lines[:, 1], lines[:, 2], lines[:, 3])
    # for line in lines:
    #     ax.plot3D([line[1], lines[int(line[5]), 1]], [line[2], lines[int(line[5]), 2]], [line[3], lines[int(line[5]), 3]])
    #
    # plt.show()
    ############################################################################
