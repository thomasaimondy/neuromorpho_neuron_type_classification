打开Multi_train.py, 修改trnn_datadir, cnn_datadir及超参数并运行
trnn_datadir trnn模型的输入数据(swc)
cnn_datadir cnn模型的输入数据(png)
trnn_model_filename = '../TRNN/model_v0'  # 训练好的trnn模型的文件名
cnn_model_filename = '../CNN/model'    # 训练好的cnn模型的文件名
maxlen_of_tree = 300  #
lr = 1e-4   # fc层的学习率
epochs = 20    # 训练的epoch数
batch_size = 100
trained = False # 是否使用训练好的fc层
fc_filename = 'fc'   # 读取和保存fc层模型的文件名