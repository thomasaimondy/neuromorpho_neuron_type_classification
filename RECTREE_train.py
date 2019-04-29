from RECTREE_input import RECTREEdataset
from RECTREE_model import naiveRNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim


# parameters
batch_size = 1000
n_hidden = 128
lr = 0.01
num_classes = 5
epochs = 16
maxlen_of_tree = 300 # 超过maxlen的样本暂时不考虑，因为太耗时了。

trainset = RECTREEdataset(dataset_dir=r'train_data', maxlen_of_tree=maxlen_of_tree)
train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
testset = RECTREEdataset(dataset_dir=r'test_data', maxlen_of_tree=maxlen_of_tree)
test_loader = DataLoader(testset, shuffle=True, batch_size=batch_size)

num_data, maxlen_of_tree, len_of_line = trainset.num_data, trainset.maxlen_of_tree, trainset.len_of_line
X, y, tree_len = iter(train_loader).next()

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


def train(model, batch_size, maxlen_of_tree, len_of_line, num_classes, epochs=10, lr=0.001):
    train_on_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1/11, 1/34, 1/10, 1/10, 1/10]))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_train_acc = [1.0 / num_classes]
    all_test_acc = [1.0 / num_classes]

    for epoch in range(epochs):
        train_running_loss = 0.0
        train_acc = 0.0
        epoch_train_acc = 0.0
        model.train()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            data, labels, tree_len = sample
            data, labels, tree_len = sort_sample(data, labels, tree_len)
            if data.shape[0] < batch_size:
                continue


            data = pack_batch(data, tree_len)
            if (train_on_gpu): inputs, labels = data.cuda(), labels.cuda()


            outputs = model(data.float())
            loss = criterion(outputs, labels.long())          # labels不用onehot吗
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, batch_size)
            # print(train_acc)

            try:
                epoch_loss = 100 * train_running_loss / i
                epoch_train_acc = train_acc / i
                all_train_acc.append(epoch_train_acc)
            except ZeroDivisionError:
                continue
            print('batch ', i, ' 前i_batch正确率', epoch_train_acc)
        print('训练正确率', epoch_train_acc)
        print('训练Loss', epoch_loss)

        test_running_loss = 0.0
        test_acc = 0.0
        epoch_test_acc = 0.0
        model.eval()

        for i, sample in enumerate(test_loader):
            model.hidden = model.init_hidden()
            data, labels, tree_len = sample
            data, labels, tree_len = sort_sample(data, labels, tree_len)
            if data.shape[0] < batch_size:
                continue


            data = pack_batch(data, tree_len)
            if (train_on_gpu): inputs, labels = data.cuda(), labels.cuda()


            outputs = model(data.float())
            loss = criterion(outputs, labels.long())          # labels不用onehot

            test_running_loss += loss.detach().item()
            test_acc += get_accuracy(outputs, labels, batch_size)
            # print(train_acc)

            try:
                epoch_loss = 100 * test_running_loss / i
                epoch_test_acc = test_acc / i
                all_test_acc.append(epoch_test_acc)
            except ZeroDivisionError:
                continue
        print('测试正确率', epoch_test_acc)


    return all_train_acc, all_test_acc

if __name__ == '__main__':
    model = naiveRNN(batch_size, len_of_line, n_hidden, num_classes)
    if torch.cuda.is_available(): model.cuda()
    print(model)
    all_train_acc, all_test_acc = train(model, batch_size, maxlen_of_tree, len_of_line, num_classes, epochs=epochs, lr=lr)
