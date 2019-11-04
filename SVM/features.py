# -- coding: utf-8 --
import pickle


#2class
data_dir = {'train': ('train/interneuron', 'train/principal cell'),
            'test': ('test/interneuron', 'test/principal cell')}

num_classes = len(data_dir['train'])

with open('features', 'rb')as f:
    features = pickle.load(f)
traindata = [[] for c in range(num_classes)]
testdata = [[] for c in range(num_classes)]

for phase in data_dir:
    for c in range(len(data_dir[phase])):
        path = data_dir[phase][c].split('/')
        feature = features
        for p in path:
            feature = feature[p]
            print (feature)



# for feature in features.values():
#     print(feature)

# self.key_list = []
# def get_dict_allkeys(self, dict_a):
#         """
#         多维/嵌套字典数据无限遍历，获取json返回结果的所有key值集合
#         :param dict_a:
#         :return: key_list
#         """
#         if isinstance(dict_a, dict):  # 使用isinstance检测数据类型
#             for x in range(len(dict_a)):
#                 temp_key = dict_a.keys()[x]
#                 temp_value = dict_a[temp_key]
#                 self.key_list.append(temp_key)
#                 self.get_dict_allkeys(temp_value)  # 自我调用实现无限遍历
#         elif isinstance(dict_a, list):
#             for k in dict_a:
#                 if isinstance(k, dict):
#                     for x in range(len(k)):
#                         temp_key = k.keys()[x]
#                         temp_value = k[temp_key]
#                         self.key_list.append(temp_key)
#                         self.get_dict_allkeys(temp_value)
#         return self.key_list



