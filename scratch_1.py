import MySQLdb
import pickle
import os
import shutil

db = MySQLdb.connect('localhost', 'tlk', 'h66261034')

cursor = db.cursor()
db.select_db('reconstruction')
sql = "SELECT neuron_id, Species_Name FROM neuromorpho"
cursor.execute(sql)
data = cursor.fetchall()
rat = set()
for item in data:
    if item[1] == 'rat':
        rat.add(item[0])

print(len([swc for swc in os.listdir('D:/seafile/study_cloud/Rnn_intern/original_rat_swc')]))

# d = 'D:/seafile/study_cloud/Rnn_intern/original_swc'
# for swc in os.listdir(d):
#     if int(swc.split('.')[0]) in rat:
#         shutil.copyfile(d + '/' + swc, 'D:/seafile/study_cloud/Rnn_intern/original_rat_swc/' + swc)


# 多类
# 生成 5 -> 100 科研
# 树形 分类
# 重建的神经元不完美，
# 聚类，去除错误的神经元
# 有一些三维结构， 把它转成swc 文件，txt