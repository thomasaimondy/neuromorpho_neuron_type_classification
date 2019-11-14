import pickle
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
import random

#2class
data_dir = {'train': ('train/interneuron', 'train/principal cell'),
            'test': ('test/interneuron', 'test/principal cell')}

#12class
# data_dir = {'train':
#                 ('train/principal cell/ganglion', 'train/principal cell/pyramidal',
#                 'train/principal cell/Purkinje', 'train/principal cell/granule',
#                 'train/principal cell/medium spiny', 'train/interneuron/Nitrergic',
#                 'train/interneuron/GABAergic', 'train/Glia/microglia',
#                 'train/principal cell/Parachromaffin', 'train/interneuron/basket',
#                 'train/Glia/astrocyte', 'train/sensory receptor'),
#             'test':
#                 ('test/principal cell/ganglion', 'test/principal cell/pyramidal',
#                 'test/principal cell/Purkinje', 'test/principal cell/granule',
#                 'test/principal cell/medium spiny', 'test/interneuron/Nitrergic',
#                 'test/interneuron/GABAergic', 'test/Glia/microglia',
#                 'test/principal cell/Parachromaffin', 'test/interneuron/basket',
#                 'test/Glia/astrocyte', 'test/sensory receptor')
#             }
num_classes = len(data_dir['train'])

def add_feature(feature, data):
    # 将字典feature中的所有样本向量添加到data中，并在labels中添加相应的标签，c表示类别是第c类
    for k in feature:
        if isinstance(feature[k], dict):
            add_feature(feature[k], data)
        else:
            data.append(feature[k])


def read_data(data_dir):
    with open('features', 'rb')as f:
        features = pickle.load(f)
    traindata = [[] for c in range(num_classes)]
    testdata = [[] for c in range(num_classes)]

    for phase in data_dir:
        for c in range(len(data_dir[phase])):
            path = data_dir[phase][c].split('/')
            feature = features
            for p in path:
                feature = feature[p]
            if phase == 'train':
                add_feature(feature, traindata[c])
            else:
                add_feature(feature, testdata[c])
    # 样本均衡
    m = max([len(fi) for fi in traindata])
    for fi in traindata:
        if len(fi) == 0:
            raise ValueError('some classes are empty')
        while len(fi) < m:
            extra = random.sample(fi, 1)
            fi.extend(extra)
    print([len(fi) for fi in traindata])

    trainlabels = [[c for _ in range(len(traindata[c]))] for c in range(num_classes)]
    testlabels = [[c for _ in range(len(testdata[c]))] for c in range(num_classes)]

    # flatten
    train_labels = []
    test_labels = []
    train_data = []
    test_data = []
    for i in range(num_classes):
        train_labels.extend(trainlabels[i])
        test_labels.extend(testlabels[i])
        train_data.extend(traindata[i])
        test_data.extend(testdata[i])

    # shuffle
    trainset = list(zip(train_data, train_labels))
    testset = list(zip(test_data, test_labels))
    random.shuffle(trainset)
    random.shuffle(testset)
    traindata, trainlabels = zip(*trainset)
    testdata, testlabels = zip(*testset)


    trainlabels, testlabels, traindata, testdata = np.array(trainlabels), np.array(testlabels), np.array(traindata), np.array(testdata)
    # print(train_data)
    # print(trainlabels.shape, testlabels.shape, traindata.shape, testdata.shape)

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

    #print (traindata[:, 0])
    #print (testdata[:, 0])
    #print (traindata[:, 0].shape, testdata[:, 0].shape)
    #print (type(testdata[:, 0]))


    ndata = [0 for c in range(num_classes)]
    for item in trainlabels:
        ndata[item] += 1
    print('各类样本数目', ndata)

    clf = SVC(C=1)
    # 因为数据集太大，只选取一部分进行训练
    clf.fit(traindata[0:50000], trainlabels[0:50000])
    train_output = clf.predict(traindata)


    test_output = clf.predict(testdata)



    print('train:', get_accuracy(train_output, trainlabels))
    print('test:', get_accuracy(test_output, testlabels))
