打开RNN_train.py, 修改data_dir及超参数并运行

batch_size = 200
n_hidden = 128  # RNN的隐层神经元个数（两层个数相同）
lr = 1e-4  # 学习率
num_classes = len(data_dir['train'])
epochs = 100
maxlen_of_tree = 1500  # 超过maxlen的样本暂时不考虑，因为太耗时了。maxlen_of_tree可以取3000以上包含绝大部分神经元如果不嫌慢
in_dim = 7   # 输入数据的维度，即swc文件每一行的宽度
trained = False   # 是否使用训练好的模型
model_filename = 'model_12class_v0' # 读取和保存模型的文件名