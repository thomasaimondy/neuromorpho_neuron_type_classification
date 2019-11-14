### Neuron Type Classification
Author, Tielin Zhang, Yue Zhang, Likai Tang in CASIA. 
tielin.zhang@ia.ac.cn, http://bii.ia.ac.cn/~tielin.zhang

# data introduction
## the structure of swc-format file.
`
 0 [0.0, 0.0, 0.0] 1 -1 7.64 0 0.0 0.0  
 1 [6.54, 3.93, 0.0] 1 0 7.64 -1 7.63 0.0  
 2 [-6.54, -3.93, 0.0] 1 0 7.64 -1 7.63 0.0  
 3 [-4.89, 11.54, -0.27] 4 0 1.09 1 12.54 71.13  
 4 [2.08, -13.49, 6.19] 3 0 0.54 1 14.99 66.08  
 ……    
`  
the structure of each line for swc file:
``
 ['id', 'P', 'type', 'parent', 'width', 'branch_level', 'path_length', 'degree']，其中,  
 id: ID of the node；  
 P：the 3d position, the soma is the (0,0,0);  
type: 1-core, 2-axon, 3-the end of the dendritic, 4-apical dendrite  
parent: father id of point. 
width: the diameter of the neuron.
branch_level：the level of node，-1-end point  
path_length：the distance from point to the center node.
degree：the angle of two branches.
```

In swc_data folder:

swc_v0   从网站上爬下来的原始数据  
swc_v1   利用软件修复的第一版数据  
swc_v3   利用软件修复的最终版本数据  
swc_fake   利用软件生成的虚拟神经元数据  
## png类型数据
在png_data文件夹
png_v0   从网站上爬下来的原始神经元图片数据  
png_v1   从swc_v1生成的图片数据  
png_fake  利用软件生成的虚拟神经元数据  
png_r    resampled_rat_img, 经过重采样的神经元图片，处理后的神经元看起来结构更加简单
png_fc   fixed_coordinate_rat_img, 固定了xy坐标轴的神经元图片

##  数据目录格式
swc 与 png 的目录格式类似  
如swc_v0 下有两个文件夹train, test  
train下有5个文件夹，如principal cell， 每个文件夹中包含prinary_cell_class 相同的一大类神经元  
principal cell下有若干文件夹， 如pyrimidal, 每个文件夹中包含secondary_cell_class 相同的一类神经元  
pyrimidal下有若干文件夹， 如Not reported, 每个文件夹中包含teritary_cell_class 相同的一类神经元  
Not reported下有若干swc文件  
##划分数据集
将所有的swc或png文件放在同一文件夹，运行scratch.py文件，通过修改最后两行的divide_dataset('./png_data/v0', './png_data/png_v0', train_id)，中间两个路径参数，来将数据集划分成训练集和测试集，以及分好大类和子类，第一个路径是划分前的文件夹，第二个路径是划分后的文件夹。

# 运行说明
## 环境
Ubuntu 16.04.6 LTS 
python3.5.2
所需的python包可以通过运行pip install -r requirements.txt来安装
## 结构
共有4个模型及其相应的输入和训练代码，包括RNN, TRNN, CNN, Multi, 分别放在对应目录下。另外还有SVM模型
每个模型包括_input, _model, _train三个代码文件。（Multi 和CNN没有_model文件）还有若干预训练好的模型文件
## _input 
_input 文件包含一个Dataset类，从torch.utils.data.Dataset继承，进行数据读取和预处理。
_input 文件中在list_file函数中考虑了样本均衡的问题，简单的把所有的类的样本数目都扩充到一样
## _model
_model文件包括构建模型所需的代码，也就是包括一个torch.nn.Module的子类
## _train
_train文件包含对数据进行训练或测试的代码，所需参数和数据目录放在文件开头部分，
如果要使用预训练好的模型，将trained参数改为True, 并将model_filename参数改为对应的模型文件名
## 运行
需要训练哪个模型，则打开相应的_train 文件，根据分两类或12类注释掉对应的另一部分，修改数据目录data_dir和超参数并运行，
data_dir的格式请参照原来的修改。如果使用新数据，需要运行scratch.py预先划分训练集和测试集。并且按类放置神经元。训练完成后，会在CNN_train文件夹下自动生成对应的模型文件。
## cuda
如果服务器上有多块GPU，使用os.environ指定GPU。CNN模型中使用了nn.DataParallel来使用四块gpu同时训练，
但是如果服务器上没有多块gpu可能会报错。

# 模型说明
## RNN 
使用两层lstm，并在后面加一层全连接层，参见structure.pptx
## TRNN
模型参见 Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks论文，
但比原论文的模型要简单。模型由两层lstmcell组成，按照从叶节点向树根节点的顺序回卷，后面再加一个全连接层, 参见structure.pptx
## CNN 
使用预训练好的resnet18，并将最后一层全连接层换掉来适应类别数。训练时以较小学习率训练前面各层，以较大学习率训练最后一层全连接层。
## Multi
Multi导入预训练好的trnn_model(不包括最后一层全连接)和cnn_model(不包括最后一层全连接)，也就是提取两部分的特征向量，
将这两部分的输出的特征向量拼接在一起，另外再加一层全连接。训练时只训练这最后一层全连接。
## SVM
从sql中选取了19个和形态学相关的特征，保存在features文件中(pickle形式，参见structure.pptx),使用svm分类（sklearn.SVC)
SVM模型中没有考虑样本均衡的问题
19个维度包括:
Total_Length, Number_of_Bifurcations, Fractal_Dimension, Number_of_Stems, Number_of_Branches, Total_Surface,
Max_Branch_Order, Average_Rall_s_Ratio, Max_Euclidean_Distance, Partition_Asymmetry, Overall_Depth, Soma_Surface,
Overall_Height, Average_Diameter, Overall_Width, Total_Volume, Average_Bifurcation_Angle_Remote,
Average_Bifurcation_Angle_Local, Max_Path_Distance



