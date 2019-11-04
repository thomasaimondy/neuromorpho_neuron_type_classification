# -- coding: utf-8 --
from RNN_input import RNNdataset
from RNN_model import naiveRNN, naiveLSTM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#
# # #2class
# data_dir = {'train': ('../swc_data/swc_v0/train/interneuron', '../swc_data/swc_v0/train/principal cell'),
#             'test': ('../swc_data/swc_v0/test/interneuron', '../swc_data/swc_v0/test/principal cell')}

# 5 classes
# data_dir = {'train': ('../swc_data/swc_v3/train/interneuron', '../swc_data/swc_v3/train/principal cell/pyramidal', '../swc_data/swc_v3/train/Glia', '../swc_data/swc_v3/train/Not reported', '../swc_data/swc_v3/train/sensory receptor'),
#              'test': ('../swc_data/swc_v3/test/interneuron', '../swc_data/swc_v3/test/principal cell/pyramidal', '../swc_data/swc_v3/test/Glia', '../swc_data/swc_v3/test/Not reported', '../swc_data/swc_v3/test/sensory receptor')
#              }

#12 classes: all classes more than 300

path = '../swc_data/swc_v3/'
data_dir = {'train':
                (path + 'train/principal cell/ganglion', path + 'train/principal cell/pyramidal',
                path + 'train/principal cell/Purkinje', path + 'train/principal cell/granule',
                path + 'train/principal cell/medium spiny', path + 'train/interneuron/Nitrergic',
                path +'train/interneuron/GABAergic', path + 'train/Glia/microglia',
                path + 'train/principal cell/Parachromaffin', path + 'train/interneuron/basket',
                path + 'train/Glia/astrocyte', path + 'train/sensory receptor'),
            'test':
                (path + 'test/principal cell/ganglion', path + 'test/principal cell/pyramidal',
                path + 'test/principal cell/Purkinje', path + 'test/principal cell/granule',
                path + 'test/principal cell/medium spiny', path + 'test/interneuron/Nitrergic',
                path + 'test/interneuron/GABAergic', path + 'test/Glia/microglia',
                path + 'test/principal cell/Parachromaffin', path + 'test/interneuron/basket',
                path + 'test/Glia/astrocyte', path + 'test/sensory receptor')
            }


# parameters
batch_size = 1
num_classes = len(data_dir['train'])
maxlen_of_tree = 1500 # 超过maxlen的样本暂时不考虑，因为太耗时了。v0是1500，v3是300
in_dim = 7
trained = True
model_filename = 'model_12class_v3'

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

model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.CrossEntropyLoss()
# best_epoch_accuracy = 0

for phase in ['train', 'test']:
    model.eval()
    if phase == 'test':
        loader = testloader
    if phase == 'train':
        loader = trainloader
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        data, labels, tree_len = batch
        data, labels, tree_len = sort_sample(data, labels, tree_len)
        if data.shape[0] < batch_size:
            continue
        if cudaFlag:
            data, labels, tree_len = data.cuda(), labels.cuda(), tree_len.cuda()

        outputs = model([data.float(), tree_len])
        predict = torch.max(outputs, dim=1)[1]
        print('predict', predict.data.cpu().numpy()[0])
        print('labels:', labels.data.cpu().numpy()[0])
        print('i_', i)

        rexcel = open_workbook("predict_12class.xls")  # 用wlrd提供的方法读取一个excel文件
        rows = rexcel.sheets()[0].nrows  # 用wlrd提供的方法获得现在已有的行数
        excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
        table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet
        table.write(i, 0, str(labels.data.cpu().numpy()[0]))  # xlwt对象的写方法，参数分别是行、列、值
        table.write(i, 1, str(predict.data.cpu().numpy()[0]))  # xlwt对象的写方法，参数分别是行、列、值
        excel.save("predict_12class.xls")  # xlwt对象的保存方法，这时便覆盖掉了原来的excel

# 83.0% is the best,  81.69% in model_5class(v1)
# 81.87% is the best,  81.81% in model_12class(v1)
# 86.11% is the best, 85.42% in model_v0, but swc_v0 has been renewed
# 86.88% is the best, 86.75% in model_v1

# 是否需要误差矩阵？
