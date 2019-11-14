from RNN_input import RNNdataset
from RNN_model import naiveRNN, naiveLSTM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 5 classes
# data_dir = {'train': ('../swc_data/swc_v3/train/interneuron', '../swc_data/swc_v3/train/principal cell/pyramidal', '../swc_data/swc_v3/train/Glia', '../swc_data/swc_v3/train/Not reported', '../swc_data/swc_v3/train/sensory receptor'),
#              'test': ('../swc_data/swc_v3/test/interneuron', '../swc_data/swc_v3/test/principal cell/pyramidal', '../swc_data/swc_v3/test/Glia', '../swc_data/swc_v3/test/Not reported', '../swc_data/swc_v3/test/sensory receptor')
#              }

#12 classes: all classes more than 300
data_dir = {'train':
                ('../swc_data/swc_v0/train/principal cell/ganglion', '../swc_data/swc_v0/train/principal cell/pyramidal',
                '../swc_data/swc_v0/train/principal cell/Purkinje', '../swc_data/swc_v0/train/principal cell/granule',
                '../swc_data/swc_v0/train/principal cell/medium spiny', '../swc_data/swc_v0/train/interneuron/Nitrergic',
                '../swc_data/swc_v0/train/interneuron/GABAergic', '../swc_data/swc_v0/train/Glia/microglia',
                '../swc_data/swc_v0/train/principal cell/Parachromaffin', '../swc_data/swc_v0/train/interneuron/basket',
                '../swc_data/swc_v0/train/Glia/astrocyte', '../swc_data/swc_v0/train/sensory receptor'),
            'test':
                ('../swc_data/swc_v0/test/principal cell/ganglion', '../swc_data/swc_v0/test/principal cell/pyramidal',
                '../swc_data/swc_v0/test/principal cell/Purkinje', '../swc_data/swc_v0/test/principal cell/granule',
                '../swc_data/swc_v0/test/principal cell/medium spiny', '../swc_data/swc_v0/test/interneuron/Nitrergic',
                '../swc_data/swc_v0/test/interneuron/GABAergic', '../swc_data/swc_v0/test/Glia/microglia',
                '../swc_data/swc_v0/test/principal cell/Parachromaffin', '../swc_data/swc_v0/test/interneuron/basket',
                '../swc_data/swc_v0/test/Glia/astrocyte', '../swc_data/swc_v0/test/sensory receptor')
            }

# parameters
batch_size = 200
n_hidden = 128
lr = 1e-4
num_classes = len(data_dir['train'])
epochs = 100
maxlen_of_tree = 1500  # 超过maxlen的样本暂时不考虑，因为太耗时了。
in_dim = 7
trained = False
model_filename = 'model_12class_v0'

trainset = RNNdataset('train', data_dir, maxlen_of_tree=maxlen_of_tree)
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testset = RNNdataset('test', data_dir, maxlen_of_tree=maxlen_of_tree)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)


def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def pack_batch(data, tree_len):
    # print(tree_len)
    pack = nn.utils.rnn.pack_padded_sequence(data, tree_len, batch_first=True)
    return pack

def sort_sample(data, label, tree_len):
    # 将样本按照序列长短排序以符合pad要求
    a = [[data[i], label[i], tree_len[i]] for i, _ in enumerate(tree_len)]
    a = sorted(a, key=lambda x: x[2], reverse=True)
    data = torch.stack([x[0] for x in a])
    label = torch.Tensor([x[1] for x in a])
    tree_len = torch.Tensor([x[2] for x in a])
    return data, label, tree_len


cudaFlag = torch.cuda.is_available()
if trained:
    model = torch.load(model_filename)
else:
    model = naiveLSTM(batch_size=batch_size, input_size=in_dim, n_hidden=n_hidden, num_classes=num_classes)

if cudaFlag: model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
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
            data, labels, tree_len = batch
            data, labels, tree_len = sort_sample(data, labels, tree_len)
            if data.shape[0] < batch_size:
                continue
            if cudaFlag:
                data, labels, tree_len = data.cuda(), labels.cuda(), tree_len.cuda()

            outputs = model([data.float(), tree_len])
            accuracy = get_accuracy(outputs, labels, batch_size)
            loss = criterion(outputs, labels.long())
            print('batch ', i, accuracy, loss)
            epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        print(phase, epoch_accuracy)
    torch.save(model, model_filename)
    # 83.0% is the best,  81.69% in model_5class(v1)
    # 81.87% is the best,  81.81% in model_12class(v1)
    # 86.11% is the best, 85.42% in model_v0, but swc_v0 has been renewed
    # 86.88% is the best, 86.75% in model_v1

    # 是否需要误差矩阵？
