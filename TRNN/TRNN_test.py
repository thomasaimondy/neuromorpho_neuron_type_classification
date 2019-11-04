# -- coding: utf-8 --
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from CNN_input import neuronSet
from torch.utils.data import DataLoader
import os
import numpy
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy
import os

#多块使用逗号隔开
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# torch.cuda.set_device(1)  # id=0, 1, 2 等

# # 2 class
# data_dir = {'train': ('../png_data/png_v1/train/interneuron', '../png_data/png_v1/train/principal cell'),
#             'test': ('../png_data/png_v1/test/interneuron', '../png_data/png_v1/test/principal cell')}


# 12 classes: all classes more than 300
path = '../png_data/swc_v3/'
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


num_classes = len(data_dir['train'])
cudaFlag = torch.cuda.is_available()

# parameters
batch_size = 1
trained = True
model_filename = 'model_12class_v3'

trainset = neuronSet('train', data_dir=data_dir)
testset = neuronSet('test', data_dir=data_dir)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

if trained:
    model = torch.load(model_filename)
print(model)
# fc_parameters = model.fc.parameters()
# l = list(map(id, model.fc.parameters()))
conv_parameters = (parameter for parameter in model.parameters() if id(parameter) not in l)

if cudaFlag:
    model.cuda()
#     model = nn.DataParallel(model, device_ids=[1])

def my_forward(model, x):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    mo = nn.Sequential(*list(model.children())[:-1])
    feature = mo(x)
    feature = feature.view(x.size(0),-1)
    output = model.fc(feature)
    return feature

def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

for phase in ['test','train']:
    model.eval()
    if phase == 'test':
        loader = testloader
    if phase == 'train':
        loader = trainloader
    for i, batch in enumerate(loader):
        data, labels = batch
        if data.shape[0] < batch_size:
            continue
        data, labels = data.cuda(), labels.cuda()
        outputs = model(data)
        print('outputs:', outputs)
        predict = torch.max(outputs, dim=1)[1]
        # print('predict', predict)
        # print('labels:', labels)
        print('predict', predict.data.cpu().numpy()[0])
        print('labels:', labels.data.cpu().numpy()[0])
        myfeature = my_forward(model, data)
        print('myfeature.shape_', myfeature.shape)
        print('myfeature_', str(myfeature[0, :].tolist()))
        print('i_', i)


        rexcel = open_workbook("predict_12class.xls")  # 用wlrd提供的方法读取一个excel文件
        rows = rexcel.sheets()[0].nrows  # 用wlrd提供的方法获得现在已有的行数
        excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
        table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet
        table.write(i, 0, str(labels.data.cpu().numpy()[0]))  # xlwt对象的写方法，参数分别是行、列、值
        table.write(i, 1, str(predict.data.cpu().numpy()[0]))  # xlwt对象的写方法，参数分别是行、列、值
        excel.save("predict_12class.xls")  # xlwt对象的保存方法，这时便覆盖掉了原来的excel
        # table.close()
        # excel.close()