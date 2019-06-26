'''
to generate two types of tree
each line = [id, x, y, z, father_id]
# 第一类小而分枝多，　第二类大而分支少
'''

import numpy as np
import math

def get_nearest(p, tree):
    m = 1e300
    nearest = 0
    for i, line in enumerate(tree):
        if i == p:
            continue
        if math.isclose(tree[i, 0], 0) or math.isclose(tree[i, 0], 1) or tree[i, 0] > 1:
            # 判断i点是否已经在树上
            continue
        if np.linalg.norm(tree[i, 1:4] - tree[p, 1:4]) < m:
            nearest = i
            m = np.linalg.norm(tree[i, 1:4] - tree[p, 1:4])
    return nearest


def add_node(p, tree):
    c = get_nearest(p, tree)
    tree[c, 4] = p   # 修改父节点
    tree[c, 0] = c   # 将自己的id 修改对
    return c

def generate_tree1(tree_len, branch_num, x_size, y_size, z_size):
    tree = np.random.random((tree_len, 5))
    tree[0] = np.array([0, 0, 0, 0, 0])      # root

    for l1 in range(2):
        a = add_node(0, tree)
        for l2 in range(3):
            b = add_node(a, tree)
            for l3 in range(3):
                c = add_node(b, tree)
    tree[:, 1] = tree[:, 1] * x_size
    tree[:, 2] = tree[:, 2] * y_size
    tree[:, 3] = tree[:, 3] * z_size
    return tree

def save_swc(filename, t):
    with open(filename, 'wb') as f:
        for i in range(t.shape[0]):
            for j in range(5):
                f.write(str(t[i, j]).encode('utf-8'))
                f.write(' '.encode('utf-8'))
            f.write('\n'.encode('utf-8'))

if __name__ == '__main__':
    for i in range(100):
        t = generate_tree1(10, 1, 1, 2, 2)
        filename = 'class_1/' + str(i) + '.swc'
        save_swc(filename, t)

