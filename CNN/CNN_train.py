import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from CNN_input import neuronSet
from torch.utils.data import DataLoader

#2class
# data_dir = {'train': ('../png_data/png_v0/train/interneuron', '../png_data/png_v0/train/principal cell'),
#             'test': ('../png_data/png_v0/test/interneuron', '../png_data/png_v0/test/principal cell')}
#12class
data_dir = {'train':
                ('../png_data/png_r/train/principal cell/ganglion', '../png_data/png_r/train/principal cell/pyramidal',
                '../png_data/png_r/train/principal cell/Purkinje', '../png_data/png_r/train/principal cell/granule',
                '../png_data/png_r/train/principal cell/medium spiny', '../png_data/png_r/train/interneuron/Nitrergic',
                '../png_data/png_r/train/interneuron/GABAergic', '../png_data/png_r/train/Glia/microglia',
                '../png_data/png_r/train/principal cell/Parachromaffin', '../png_data/png_r/train/interneuron/basket',
                '../png_data/png_r/train/Glia/astrocyte', '../png_data/png_r/train/sensory receptor'),
            'test':
                ('../png_data/png_r/test/principal cell/ganglion', '../png_data/png_r/test/principal cell/pyramidal',
                '../png_data/png_r/test/principal cell/Purkinje', '../png_data/png_r/test/principal cell/granule',
                '../png_data/png_r/test/principal cell/medium spiny', '../png_data/png_r/test/interneuron/Nitrergic',
                '../png_data/png_r/test/interneuron/GABAergic', '../png_data/png_r/test/Glia/microglia',
                '../png_data/png_r/test/principal cell/Parachromaffin', '../png_data/png_r/test/interneuron/basket',
                '../png_data/png_r/test/Glia/astrocyte', '../png_data/png_r/test/sensory receptor')
            }

num_classes = len(data_dir['train'])
cudaFlag = torch.cuda.is_available()

# parameters
lr_fc = 1e-4  # 最后一层的学习率
lr_conv = 1e-5  # 前面各层的学习率
epochs = 20
batch_size = 100
trained = False
model_filename = 'model_12class_r'

trainset = neuronSet('train', data_dir=data_dir)
testset = neuronSet('test', data_dir=data_dir)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

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
    model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

def get_accuracy(logit, target, batch_size):
    # print(torch.max(logit, dim=1)[1])
    corrects = (torch.max(logit, dim=1)[1].view(target.size()).data == target.long().data).sum() #虽然没有经过softmax,但是softmax是单调的
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()

# 以较大学习率训练输出层，以较小学习率训练前面各层
optimizer = optim.Adam([{'params': fc_parameters, 'lr': lr_fc}, {'params': conv_parameters, 'lr': lr_conv}])
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
            data, labels = batch
            if data.shape[0] < batch_size:
                continue
            if cudaFlag:
                data, labels = data.cuda(), labels.cuda()

            outputs = model(data)
            accuracy = get_accuracy(outputs, labels, batch_size)
            loss = criterion(outputs, labels.long())
            print('batch_', i, accuracy, loss)
            epoch_accuracy = (epoch_accuracy * i + accuracy) / (i + 1)
            if phase == 'train':
                loss.backward()
                optimizer.step()
        print(phase, 'epoch_', epoch, epoch_accuracy)
    torch.save(model.module, model_filename)
    # 86.49% in 'model', the best was 87.87%



