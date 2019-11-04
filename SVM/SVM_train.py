import pickle
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
import random

#2class
# data_dir = {'train': ('train/interneuron', 'train/principal cell'),
#             'test': ('test/interneuron', 'test/principal cell')}

#12class
data_dir = {'train':
                ('train/principal cell/ganglion', 'train/principal cell/pyramidal',
                'train/principal cell/Purkinje', 'train/principal cell/granule',
                'train/principal cell/medium spiny', 'train/interneuron/Nitrergic',
                'train/interneuron/GABAergic', 'train/Glia/microglia',
                'train/principal cell/Parachromaffin', 'train/interneuron/basket',
                'train/Glia/astrocyte', 'train/sensory receptor'),
            'test':
                ('test/principal cell/ganglion', 'test/principal cell/pyramidal',
                'test/principal cell/Purkinje', 'test/principal cell/granule',
                'test/principal cell/medium spiny', 'test/interneuron/Nitrergic',
                'test/interneuron/GABAergic', 'test/Glia/microglia',
                'test/principal cell/Parachromaffin', 'test/interneuron/basket',
                'test/Glia/astrocyte', 'test/sensory receptor')
            }


def add_feature(feature, data, labels, c):
    # 将字典feature中的所有样本向量添加到data中，并在labels中添加相应的标签，c表示类别是第c类
    for k in feature:
        if isinstance(feature[k], dict):
            add_feature(feature[k], data, labels, c)
        else:
            data.append(feature[k])
            labels.append(c)


def read_data(data_dir):
    with open('features', 'rb')as f:
        features = pickle.load(f)

    traindata = []
    testdata = []
    trainlabels = []
    testlabels = []

    for phase in data_dir:
        for c in range(len(data_dir[phase])):
            path = data_dir[phase][c].split('/')
            feature = features
            for p in path:
                feature = feature[p]
            if phase == 'train':
                add_feature(feature, traindata, trainlabels, c)
            else:
                add_feature(feature, testdata, testlabels, c)


    # shuffle
    trainset = list(zip(traindata, trainlabels))
    testset = list(zip(testdata, testlabels))
    random.shuffle(trainset)
    random.shuffle(testset)
    traindata, trainlabels = zip(*trainset)
    testdata, testlabels = zip(*testset)


    trainlabels, testlabels, traindata, testdata = np.array(trainlabels), np.array(testlabels), np.array(traindata), np.array(testdata)
    print(trainlabels.shape, testlabels.shape, traindata.shape, testdata.shape)

    traindata, testdata = preprocessing.scale(traindata), preprocessing.scale(testdata)  # 数据标准化
    return trainlabels, testlabels, traindata, testdata

def get_accuracy(output, labels):
    right = 0
    for i in range(len(labels)):
        if output[i] == labels[i]:
            right += 1
    accuracy = right / len(labels)
    return accuracy

if __name__ == '__main__':

    trainlabels, testlabels, traindata, testdata = read_data(data_dir)

    clf = SVC(C=100)
    clf.fit(traindata, trainlabels)
    train_output = clf.predict(traindata)


    test_output = clf.predict(testdata)



    print('train:', get_accuracy(train_output, trainlabels))
    print('test:', get_accuracy(test_output, testlabels))
