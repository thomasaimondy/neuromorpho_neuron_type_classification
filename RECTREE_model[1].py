import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class naiveRNN(nn.Module):
    def __init__(self, batch_size, input_size, n_hidden, num_classes):
        super(naiveRNN, self).__init__()

        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.input_size = input_size
        self.num_classes = num_classes

        self.basic_rnn = nn.RNN(self.input_size, self.n_hidden, num_layers=2, batch_first=True)
        self.FC = nn.Linear(self.n_hidden, self.num_classes)

    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        h0 = (torch.zeros(2, self.batch_size, self.n_hidden))
        # if torch.cuda.is_available(): hidden = hidden.cuda()
        return h0

    def forward(self, X):
        # transforms X to dimensions: seq-len x batch-size x num-chars
        # X = X.permute(1, 0, 2)

        self.h0 = self.init_hidden()

        h_outs, self.hn = self.basic_rnn(X, self.h0)
        unpacked = nn.utils.rnn.pad_packed_sequence(h_outs,batch_first=True)
        # print(unpacked[0][0][-1])
        # print(self.hn[1][0])


        out = self.FC(self.hn)[1]
        # out = F.softmax(out, dim=1)    使用crossEntropyLoss使不需要softmax

        return out.view(-1, self.num_classes)  # batch_size X n_output

if __name__ == '__main__':

    def pack_batch(data, tree_len):
        # print(tree_len)
        pack = nn.utils.rnn.pack_padded_sequence(data, tree_len, batch_first=True)
        return pack


    def sort_sample(data, label, tree_len):
        a = [[data[i], label[i], tree_len[i]] for i, _ in enumerate(tree_len)]
        a = sorted(a, key=lambda x: x[2], reverse=True)
        data = torch.stack([x[0] for x in a])
        label = torch.Tensor([x[1] for x in a])
        tree_len = torch.Tensor([x[2] for x in a])
        return data, label, tree_len


    data = torch.from_numpy(np.random.random([3, 100, 10]))  # batch_size, maxlen, len_of_line
    data1 = data.clone()
    labels = torch.from_numpy(np.random.random([3]))
    tree_len = torch.from_numpy(np.array([100, 100, 100]))

    data, labels, tree_len = sort_sample(data, labels, tree_len)

    data = pack_batch(data, tree_len)

    # print(data)
    # print(labels)
    # print(tree_len)
    model = naiveRNN(3, 10, 128, 5)
    model.hidden = model.init_hidden()
    outputs0 = model(data.float())
