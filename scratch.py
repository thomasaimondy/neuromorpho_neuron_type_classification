# -- coding: utf-8 --
import os
import shutil
import pickle
import random
from TRNN_input import TRNNdataset

data_dir = {'train': ('swc_data/swc_v1/train/interneuron', 'swc_data/swc_v1/train/principal cell', 'swc_data/swc_v1/train/Glia', 'swc_data/swc_v1/train/Not reported', 'swc_data/swc_v1/train/sensory receptor'),
            'test': ('swc_data/swc_v1/test/interneuron', 'swc_data/swc_v1/test/principal cell')}

def get_id_set(data_dir, phase):
    # return all neuro_id in a given data_dir
    c = TRNNdataset(phase, data_dir=data_dir, maxlen_of_tree=1000000)
    id_set = set()
    for item in range(c.__len__()):
        id_set.add(c.getitem(item))
    print('len_id_set', len(id_set))
    return id_set


def list_data(d, dt):
    for d1 in os.listdir(d):
        # os.mkdir(dt + '/' + d1)

        for d2 in os.listdir(d + '/' + d1):
            # os.mkdir(dt + '/' + d1 + '/' + d2)
            i = 0

            for d3 in os.listdir(d + '/' + d1 + '/' + d2):
                # os.mkdir(dt + '/' + d1 + '/' + d2 + '/' + d3)


                for file in os.listdir(d + '/' + d1 + '/' + d2 + '/' + d3):
                    i += 1
                    # neuro_id = int(file.split('.')[0])
            print(d1 + '/' + d2, i)
                    # shutil.move(d + '/' + d1 + '/' + d2 + '/' + d3 + '/' + file, dt + '/' + d1 + '/' + d2 + '/' + d3 + '/' + file)

def divide_dataset(path, path_new, train_id):
    with open('nspst', 'rb') as f:
        nspst = pickle.load(f)
        nspst = [[item[0], item[1:5]] for item in nspst]
        nspst = dict(nspst)

    # 将属于小鼠的神经元移入path_new并划分训练，测试集
    os.mkdir(path_new + '/train')
    os.mkdir(path_new + '/test')
    for file in os.listdir(path):
        neuron_id = int(file.split('.')[0])
        if nspst[neuron_id][0] == 'rat':
            if neuron_id in train_id:
                shutil.copyfile(path + '/' + file, path_new + '/train/' + file)
            else:
                shutil.copyfile(path + '/' + file, path_new + '/test/' + file)

    # 将数据集按cell_type 放入不同文件夹
    it = 0
    for phase in ['train', 'test']:
        for file in os.listdir(path_new + '/' + phase):
            try:
                neuron_id = int(file.split('.')[0])
            except ValueError as e:
                print(e)
                continue
            cell_type = nspst[neuron_id][1:]
            cell_type = list(cell_type)
            for i in range(len(cell_type)):
                cell_type[i] = cell_type[i].replace('/', ' or ')

            d1 = path_new + '/' + phase + '/' + cell_type[0]
            d2 = d1 + '/' + cell_type[1]
            d3 = d2 + '/' + cell_type[2]

            if cell_type[0] not in os.listdir(path_new + '/' + phase):
                os.mkdir(d1)
            if cell_type[1] not in os.listdir(d1):
                os.mkdir(d2)
            if cell_type[2] not in os.listdir(d2):
                os.mkdir(d3)
            shutil.move(path_new + '/' + phase + '/' + file, d3 + '/' + file)
            it += 1
            print(it)
    pass

if __name__ == '__main__':
    train_id = get_id_set(data_dir, 'train')
    divide_dataset('./png_data/v0', './png_data/png_v0', train_id)

    # principal cell/ganglion 301， principal cell/pyramidal 8333， principal cell/Purkinje 405， principal cell/granule 415
    # principal cell/medium spiny 708， interneuron/Nitrergic 1718， interneuron/GABAergic 636， Glia/microglia 2931
    # principal cell/Parachromaffin 439， interneuron/basket 396， Glia/astrocyte 448
    # sensory receptor 314


    # id_set = get_id_set(data_dir, 'train')
    #
    # with open('SVM/figure', 'rb') as f:
    #     figure = pickle.load(f)
    #
    # features = {'train':{}, 'test':{}}
    #
    # for d1 in os.listdir(d):
    #     features['train'][d1.split('.')[0]] = {}
    #     features['test'][d1.split('.')[0]] = {}
    #
    #     for d2 in os.listdir(d + '/' + d1):
    #         features['train'][d1.split('.')[0]][d2.split('.')[0]] = {}
    #         features['test'][d1.split('.')[0]][d2.split('.')[0]] = {}
    #
    #         for d3 in os.listdir(d + '/' + d1 + '/' + d2):
    #             features['train'][d1.split('.')[0]][d2.split('.')[0]][d3.split('.')[0]] = {}
    #             features['test'][d1.split('.')[0]][d2.split('.')[0]][d3.split('.')[0]] = {}
    #
    #             for file in os.listdir(d + '/' + d1 + '/' + d2 + '/' + d3):
    #                 neuro_id = int(file.split('.')[0])
    #                 try:
    #                     if neuro_id in id_set:
    #                         features['train'][d1.split('.')[0]][d2.split('.')[0]][d3.split('.')[0]][neuro_id] = figure[neuro_id]
    #                     else:
    #                         print('hhh----')
    #                         features['test'][d1.split('.')[0]][d2.split('.')[0]][d3.split('.')[0]][neuro_id] = figure[neuro_id]
    #                 except KeyError:
    #                     continue
    #
    #             for file in os.listdir(dt + '/' + d1 + '/' + d2 + '/' + d3):
    #                 neuro_id = int(file.split('.')[0])
    #                 try:
    #                     if neuro_id in id_set:
    #                         features['train'][d1.split('.')[0]][d2.split('.')[0]][d3.split('.')[0]][neuro_id] = figure[neuro_id]
    #                     else:
    #                         print('hhh----')
    #                         features['test'][d1.split('.')[0]][d2.split('.')[0]][d3.split('.')[0]][neuro_id] = figure[neuro_id]
    #                 except KeyError:
    #                     continue
    #
    # i = 0
    # for phase in ['train']:
    #     for d1 in features[phase]:
    #         for d2 in features[phase][d1]:
    #             for d3 in features[phase][d1][d2]:
    #
    #                 for item in features[phase][d1][d2][d3]:
    #                     i += 1
    #                     pass
    # print(i)
    #
    #
    # with open('features', 'wb') as f:
    #     pickle.dump(features, f)
    #


