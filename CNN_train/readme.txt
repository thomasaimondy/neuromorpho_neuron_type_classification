打开CNN_train.py, 修改data_dir及超参数并运行
lr_fc = 1e-4  # 最后一层的学习率
lr_conv = 1e-5  # 前面各层的学习率
epochs = 20    # 训练的epoch数
batch_size = 100
trained = False # 是否使用训练好的模型
model_filename = 'model_12class_r' # 读取和保存模型的文件名