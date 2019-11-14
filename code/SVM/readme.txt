打开SVM_train.py, 修改data_dir并运行
如果要修改SVM的超参数， 跳到 if __name__ == '__main__': 下面， clf = SVC(...), 修改SVC参数（除了C还有其它可以改的，参见help）