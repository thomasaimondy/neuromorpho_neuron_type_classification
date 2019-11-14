from torch.utils.data import Dataset
from TRNN import TRNN_input
from CNN import CNN_input

class multiSet(Dataset):
    def __init__(self, phase, trnn_datadir, cnn_datadir, maxlen_of_tree):
        self.phase = phase
        self.TRNNdataset = TRNN_input.TRNNdataset(phase, trnn_datadir, maxlen_of_tree=maxlen_of_tree)
        self.PNGset = CNN_input.neuronSet(phase, cnn_datadir)
        self.item_list = self.get_item_list(renew=True)
        print(self.phase, len(self.item_list))

    def __getitem__(self, item):
        datum1, label1, tree_len = self.TRNNdataset.__getitem__(self.item_list[item][0])
        datum2, label2 = self.PNGset.__getitem__(self.item_list[item][1])
        if not label1 == label2:
            print('error:', label1, label2)
        return datum1, tree_len, datum2, label2
    def __len__(self):
        return len(self.item_list)

    def get_item_list(self, renew=False):
        # 将同一个样本的png格式和swc格式匹配起来
        # 返回[[样本1的swc格式在TRNNdataset中的编号，样本1的png格式在PNGset中的编号]，[样本2的swc格式在TRNNdataset中的编号，样本2的png格式在PNGset中的编号]， ... ]
        import pickle
        if not renew:
            if self.phase == 'train':
                with open('train_item_map', 'rb') as f:
                    item_map = pickle.load(f)
            else:
                with open('test_item_map', 'rb') as f:
                    item_map = pickle.load(f)
        else:
            item_map = dict()
            ad = {}
            for idex1 in range(self.TRNNdataset.__len__()):
                id1 = self.TRNNdataset.getitem(idex1)
                ad[id1] = idex1

            for idex2 in range(self.PNGset.__len__()):
                id2 = self.PNGset.getitem(idex2)
                # print('id2', id2)
                if id2 in ad:
                    # print('len', len(item_map))
                    item_map[ad[id2]] = idex2
            # print(item_map)
            with open(self.phase + '_item_map', 'wb') as f:
                pickle.dump(item_map, f)

        item_list = [[k, item_map[k]] for k in item_map]
        return item_list

if __name__ == '__main__':
    trainset = multiSet('test')