import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import torch.optim as optim


# 应该改成输入batch of tensor, 转化为batch of tree, 最后输出分类结果

class Tree(object):
    # 应该是TreeNode类
    def __init__(self, x):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.x = x
        self.state = None
        self.gold_label = None # node label for SST
        self.output = None # output node for SST

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):   # 计算从自己开始的子树大小
        try:
            return self._size
        except AttributeError:
            count = 1
            for i in range(self.num_children):
                count += self.children[i].size()
            self._size = count
            return self._size

    def depth(self):   # 计算子树高度
        try:
            return self._depth
        except AttributeError:
            count = 0
            if self.num_children>0:
                for i in range(self.num_children):
                    child_depth = self.children[i].depth()
                    if child_depth>count:
                        count = child_depth
                count += 1
            self._depth = count
            return self._depth

class naiveTreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, num_classes):
        super(naiveTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.cell = nn.LSTMCell(in_dim, mem_dim)
        self.FC = nn.Linear(mem_dim, num_classes)
        pass

    def tree_rnn(self, tree):
        child_c = torch.zeros(1, self.mem_dim)
        child_h = torch.zeros(1, self.mem_dim)
        if self.cudaFlag:
            child_c, child_h = child_c.cuda(), child_h.cuda()

        if tree.num_children != 0:
        # 如果不是叶节点

            for idx in range(tree.num_children):
                if tree.children[idx].state == None: # 如果有一个儿子的state是None， 递归到那个孩子
                    self.tree_rnn(tree.children[idx])

                child_h += tree.children[idx].state[0]
                child_c += tree.children[idx].state[1]

            # 更新state
        x = F.torch.unsqueeze(tree.x, 0)
        if self.cudaFlag:
            x = x.cuda()
        tree.state = self.cell(x, (child_h, child_c))
        return tree.state

    def forward(self, tree):
        state = self.tree_rnn(tree)
        output = self.FC(state[1])
        return output

class TreeLSTM(nn.Module):
    def __init__(self, cuda, in_dim, mem_dim, num_classes):
        super(TreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.cell1 = nn.LSTMCell(in_dim, mem_dim)
        # self.dropout = nn.Dropout(p=0)
        self.cell2 = nn.LSTMCell(mem_dim, mem_dim)
        self.FC = nn.Linear(mem_dim, num_classes)

    def tree_rnn(self, tree):
        child_c1 = torch.zeros(1, self.mem_dim)
        child_h1 = torch.zeros(1, self.mem_dim)
        child_c2 = torch.zeros(1, self.mem_dim)
        child_h2 = torch.zeros(1, self.mem_dim)
        tree.state = [0, 0, 0, 0]
        x = F.torch.unsqueeze(tree.x, 0)
        if self.cudaFlag:
            child_c1, child_h1 = child_c1.cuda(), child_h1.cuda()
            child_c2, child_h2 = child_c2.cuda(), child_h2.cuda()
            x = x.cuda()

        if tree.num_children != 0:
        # 如果不是叶节点

            for idx in range(tree.num_children):
                if tree.children[idx].state == None: # 如果有一个儿子的state是None， 递归到那个孩子
                    self.tree_rnn(tree.children[idx])

                child_h1 += tree.children[idx].state[0]
                child_c1 += tree.children[idx].state[1]
                child_h2 += tree.children[idx].state[2]
                child_c2 += tree.children[idx].state[3]


        # 更新state

        tree.state[0], tree.state[1] = self.cell1(x, (child_h1, child_c1))
        # d = self.dropout(tree.state[1])
        tree.state[2], tree.state[3] = self.cell2(tree.state[1], (child_h2, child_c2))
        return tree.state

    def forward(self, tree):
        state = self.tree_rnn(tree)
        output = self.FC(state[3])
        return output

if __name__ == '__main__':
    model = TreeLSTM(cuda=False, in_dim=10, mem_dim=128, num_classes=5)

    datum0 = Tree(torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
    datum1 = Tree(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    datum2 = Tree(torch.Tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
    datum0.add_child(datum1)
    datum0.add_child(datum2)

    datum3 = Tree(torch.Tensor([0, 0, 3, 4, 5, 6, 7, 8, 9, 0]))
    datum4 = Tree(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    datum5 = Tree(torch.Tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]))
    datum6 = Tree(torch.Tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]))
    datum3.add_child(datum4)
    datum3.add_child(datum5)
    datum4.add_child(datum6)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    trees = [datum0, datum3]
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = torch.zeros(2, 5)
        label = torch.Tensor([0, 1])
        for j in range(2):
            output = model(trees[j])
            outputs[j] = output
            print(outputs, label)

        loss = criterion(outputs, label.long())  # labels不用onehot吗
        loss.backward(retain_graph=True)
        print(loss)
        optimizer.step()

    # model = naiveRNN(1, 10, 128, 5)
    #
    # datum0 = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]).unsqueeze(0)
    # datum3 = torch.Tensor([[1, 0, 3, 4, 5, 6, 7, 0, 9, 0],[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]).unsqueeze(0)
    #
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # model.train()
    # trees = [datum0, datum3]
    # for epoch in range(100):
    #     optimizer.zero_grad()
    #     outputs = torch.zeros(2, 5)
    #     label = torch.Tensor([0, 1])
    #     for j in range(2):
    #         output = model([trees[j], torch.Tensor([3])])
    #         outputs[j] = output
    #         print(outputs, label)
    #
    #     loss = criterion(outputs, label.long())  # labels不用onehot吗
    #     print(loss)
    #     loss.backward(retain_graph=True)
    #     optimizer.step()


