import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from math import floor

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


class naiveRNN(nn.Module):
    def __init__(self, batch_size, input_size, n_hidden, num_classes):
        super(naiveRNN, self).__init__()

        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.num_classes = num_classes

        self.cnnout_size = 30
        self.cnn_1d = nn.Conv1d(self.input_size, self.cnnout_size, 3, 2, padding=1)
        self.basic_rnn = nn.RNN(self.cnnout_size, self.n_hidden, num_layers=2, batch_first=True)
        self.FC = nn.Linear(self.n_hidden, self.num_classes)

    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        h0 = (torch.zeros(2, self.batch_size, self.n_hidden))
        if torch.cuda.is_available(): h0 = h0.cuda()
        return h0

    def forward(self, X):
        # transforms X to dimensions: seq-len x batch-size x num-chars
        # X = X.permute(1, 0, 2)
        tree_len = X[1]
        tree_len = (tree_len/2).ceil().int()
        X = X[0]
        self.h0 = self.init_hidden()

        X = X.permute(0, 2, 1)
        X = self.cnn_1d(X)
        X = X.permute(0, 2, 1)

        X = pack_batch(X, tree_len)
        h_outs, self.hn = self.basic_rnn(X, self.h0)
        unpacked = nn.utils.rnn.pad_packed_sequence(h_outs,batch_first=True)

        out = self.FC(self.hn)[1]
        # out = F.softmax(out, dim=1)    使用crossEntropyLoss使不需要softmax

        return out.view(-1, self.num_classes)  # batch_size X n_output

class naiveLSTM(nn.Module):
    def __init__(self, batch_size, input_size, n_hidden, num_classes):
        super(naiveLSTM, self).__init__()

        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.num_classes = num_classes

        self.cnnout_size = 30
        self.cnn_1d = nn.Conv1d(self.input_size, self.cnnout_size, 3, 2, padding=1)
        self.basic_LSTM = nn.LSTM(self.cnnout_size, self.n_hidden, num_layers=2, batch_first=True)
        self.FC = nn.Linear(self.n_hidden, self.num_classes)

    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        h0 = (torch.zeros(2, self.batch_size, self.n_hidden))
        c0 = (torch.zeros(2, self.batch_size, self.n_hidden))
        if torch.cuda.is_available(): h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def forward(self, X):
        tree_len = X[1]
        tree_len = (tree_len/2).ceil().int()
        X = X[0]
        self.h0, self.c0 = self.init_hidden()

        X = X.permute(0, 2, 1)
        X = self.cnn_1d(X)
        X = X.permute(0, 2, 1)

        X = pack_batch(X, tree_len)
        h_outs, (self.hn, self.cn) = self.basic_LSTM(X, (self.h0, self.c0))
        unpacked = nn.utils.rnn.pad_packed_sequence(h_outs,batch_first=True)

        out = self.FC(self.hn)[1]
        # out = F.softmax(out, dim=1)    使用crossEntropyLoss使不需要softmax

        return out.view(-1, self.num_classes)  # batch_size X n_output

if __name__ == '__main__':



    data = torch.from_numpy(np.random.random([3, 100, 10]))  # batch_size, maxlen, len_of_line
    data1 = data.clone()
    labels = torch.from_numpy(np.random.random([3]))
    tree_len = torch.from_numpy(np.array([100, 80, 61]))

    data, labels, tree_len = sort_sample(data, labels, tree_len)


    model = naiveLSTM(3, 10, 128, 5)
    model.hidden = model.init_hidden()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs0 = model([data.float(), tree_len])
        labels = torch.Tensor([1,1,1])
        loss = criterion(outputs0, labels.long())
        loss.backward()
        optimizer.step()
        print(loss)