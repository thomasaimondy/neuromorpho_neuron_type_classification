import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from CNN_input import neuronSet
from torch.utils.data import DataLoader
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3,'
#2class
data_dir = {'train': ('../png_data/png_v1/train/interneuron', '../png_data/png_v1/train/principal cell'),
            'test': ('../png_data/png_v1/test/interneuron', '../png_data/png_v1/test/principal cell')}
#12class
# path = '../png_data/png_v1/'
# data_dir = {'train':
#                 (path + 'train/principal cell/ganglion', path + 'train/principal cell/pyramidal',
#                 path + 'train/principal cell/Purkinje', path + 'train/principal cell/granule',
#                 path + 'train/principal cell/medium spiny', path + 'train/interneuron/Nitrergic',
#                 path +'train/interneuron/GABAergic', path + 'train/Glia/microglia',
#                 path + 'train/principal cell/Parachromaffin', path + 'train/interneuron/basket',
#                 path + 'train/Glia/astrocyte', path + 'train/sensory receptor'),
#             'test':
#                 (path + 'test/principal cell/ganglion', path + 'test/principal cell/pyramidal',
#                 path + 'test/principal cell/Purkinje', path + 'test/principal cell/granule',
#                 path + 'test/principal cell/medium spiny', path + 'test/interneuron/Nitrergic',
#                 path + 'test/interneuron/GABAergic', path + 'test/Glia/microglia',
#                 path + 'test/principal cell/Parachromaffin', path + 'test/interneuron/basket',
#                 path + 'test/Glia/astrocyte', path + 'test/sensory receptor')
#             }

num_classes = len(data_dir['train'])
cudaFlag = torch.cuda.is_available()

# parameters
lr_fc = 1e-4  # 最后一层的学习率
lr_conv = 1e-5  # 前面各层的学习率
epochs = 20
batch_size = 300
trained = False
model_filename = 'model_2class_v1'
device_ids = [0, 1, 2, 3]
trainset = neuronSet('train', data_dir=data_dir)
testset = neuronSet('test', data_dir=data_dir)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

iter_num_list = []
loss_list = []
accuracy_list = []


if trained:
    model = torch.load(model_filename)
else:
    model = models.resnet18(pretrained=True)
    # for parameter in model.parameters():
    #     parameter.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
print(model)
fc_parameters = model.fc.parameters()
l = list(map(id, model.fc.parameters()))
conv_parameters = (parameter for parameter in model.parameters() if id(parameter) not in l)

if cudaFlag:
    #model.cuda()
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device_ids[0])
def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# 以较大学习率训练输出层，以较小学习率训练前面各层
optimizer = optim.Adam([{'params': fc_parameters, 'lr': lr_fc}, {'params': conv_parameters, 'lr': lr_conv}])
criterion = nn.CrossEntropyLoss()
best_epoch_accuracy = 0
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
            data, labels = batch
            if data.shape[0] < batch_size:
                continue
            if cudaFlag:
                data, labels = data.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])

            outputs = model(data)
            accuracy = get_accuracy(outputs, labels, batch_size)
            loss = criterion(outputs, labels.long())
            print('batch_', i, accuracy, loss)
            epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        if phase == 'test':
            # print(type(epoch))
            # print(type(loss.cpu().item()))
            # print(type(epoch_accuracy))
            iter_num_list.append(epoch)
            # loss_list.append(loss.cpu().detach().numpy())
            loss_list.append(loss.cpu().item())
            accuracy_list.append(epoch_accuracy)
            if epoch_accuracy > best_epoch_accuracy:
                best_epoch_accuracy = epoch_accuracy
        print('epoch_', epoch, phase, epoch_accuracy, 'best_accuracy_', best_epoch_accuracy)
        print('iter_num_list:', iter_num_list)
        print('loss_list:', loss_list)
        print('accuracy_list:', accuracy_list)

    torch.save(model.module, model_filename)
    # 86.49% in 'model', the best was 87.87%
