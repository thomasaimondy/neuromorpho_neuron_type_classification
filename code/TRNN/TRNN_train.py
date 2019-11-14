from TRNN.TRNN_model import naiveTreeLSTM, TreeLSTM, Tree
from TRNN.TRNN_input import TRNNdataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#2class
data_dir = {'train': ('../swc_data/swc_v3/train/interneuron', '../swc_data/swc_v3/train/principal cell'),
            'test': ('../swc_data/swc_v3/test/interneuron', '../swc_data/swc_v3/test/principal cell')}

# 5 classes
# data_dir = {'train': ('../swc_data/swc_v3/train/interneuron', '../swc_data/swc_v3/train/principal cell/pyramidal', '../swc_data/swc_v3/train/Glia', '../swc_data/swc_v3/train/Not reported', '../swc_data/swc_v3/train/sensory receptor'),
#             'test': ('../swc_data/swc_v3/test/interneuron', '../swc_data/swc_v3/test/principal cell/pyramidal', '../swc_data/swc_v3/test/Glia', '../swc_data/swc_v3/test/Not reported', '../swc_data/swc_v3/test/sensory receptor')}

# 12 classes: all classes more than 300
# data_dir = {'train':
#                 ('../swc_data/swc_v3/train/principal cell/ganglion', '../swc_data/swc_v3/train/principal cell/pyramidal',
#                 '../swc_data/swc_v3/train/principal cell/Purkinje', '../swc_data/swc_v3/train/principal cell/granule',
#                 '../swc_data/swc_v3/train/principal cell/medium spiny', '../swc_data/swc_v3/train/interneuron/Nitrergic',
#                 '../swc_data/swc_v3/train/interneuron/GABAergic', '../swc_data/swc_v3/train/Glia/microglia',
#                 '../swc_data/swc_v3/train/principal cell/Parachromaffin', '../swc_data/swc_v3/train/interneuron/basket',
#                 '../swc_data/swc_v3/train/Glia/astrocyte', '../swc_data/swc_v3/train/sensory receptor'),
#             'test':
#                 ('../swc_data/swc_v3/test/principal cell/ganglion', '../swc_data/swc_v3/test/principal cell/pyramidal',
#                 '../swc_data/swc_v3/test/principal cell/Purkinje', '../swc_data/swc_v3/test/principal cell/granule',
#                 '../swc_data/swc_v3/test/principal cell/medium spiny', '../swc_data/swc_v3/test/interneuron/Nitrergic',
#                 '../swc_data/swc_v3/test/interneuron/GABAergic', '../swc_data/swc_v3/test/Glia/microglia',
#                 '../swc_data/swc_v3/test/principal cell/Parachromaffin', '../swc_data/swc_v3/test/interneuron/basket',
#                 '../swc_data/swc_v3/test/Glia/astrocyte', '../swc_data/swc_v3/test/sensory receptor')
#             }

# parameters
in_dim = 7
batch_size = 100
n_hidden = 128
lr = 1e-4
num_classes = len(data_dir['train'])
epochs = 100
maxlen_of_tree = 300
trained = False
model_filename = 'model_2class_v3'


trainset = TRNNdataset('train', data_dir=data_dir, maxlen_of_tree=maxlen_of_tree)
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testset = TRNNdataset('test', data_dir=data_dir, maxlen_of_tree=maxlen_of_tree)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

def lines2tree(lines):
    # lines[0][5]不一定是-1
    # 寻找根节点
    i2p = dict([[int(p[0]), int(p[6])] for p in lines])  # 根据行数i返回其parent的行数p
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
        try:
            if int(i2p[i + 1]) == -1:  # 如果是根节点， 跳过
                continue
        except KeyError:
            print(lines)
        if i + 1 not in p2t:  # 如果第i行还没有成为树节点
            add_node(i + 1, line)  # 将它父亲所在的行生成节点并加入树

    # print(root.depth())
    # print(root.size())
    # print(lines.shape)
    # if root.size() != lines.shape[0]:
    #     raise KeyError('')
    return root

def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

if __name__ == '__main__':
    cudaFlag = torch.cuda.is_available()
    if trained:
        model = torch.load(model_filename)
    else:
        model = TreeLSTM(cuda=cudaFlag, in_dim=in_dim, mem_dim=128, num_classes=num_classes)
    model.cudaFlag = cudaFlag
    if (cudaFlag): model.cuda()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for phase in ['train', 'test']:
            epoch_accuracy = 0
            if phase == 'train':
                model.train()
                loader = trainloader
            else:
                model.eval()
                loader = testloader
            for i, batch in enumerate(loader):
                optimizer.zero_grad()
                outputs = torch.zeros(batch_size, num_classes)
                data, labels, tree_len = batch
                if data.shape[0] < batch_size:
                    continue
                if cudaFlag:
                    data, labels, tree_len, outputs = data.cuda(), labels.cuda(), tree_len.cuda(), outputs.cuda()

                labels = labels.resize(batch_size)
                for j in range(batch_size):
                    output = model(lines2tree((data[j][0:tree_len[j]]).float()))
                    outputs[j] = output
                accuracy = get_accuracy(outputs, labels, batch_size)
                loss = criterion(outputs, labels.long())
                print('batch ', i, accuracy, loss)
                epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            print(phase, epoch_accuracy)
        torch.save(model, model_filename)
        # 85.87% in 'model_v0', the best was 87.76%, but swc has been updated
        # 87.67% in 'model_v1', the best was 87.90%




    # 即使学习率wei0 ， 也会在若干次迭代之后陷入全部为nan， 而第一次不全部为nan， 可能和 -- 有关系
    # 也可能和Tree 不对有关系
    # 学习率调成0 似乎还是会学习？？

    # 文件名和标签的对应关系有错吗？！！！


    # 改标签，只保留小鼠，从减少样本改为复制样本 -> 正确率大幅提升
