import urllib
import urllib.request
import json
import os
import threadpool
import random
import re
from bs4 import BeautifulSoup
from unidecode import unidecode
from queue import Queue
import pickle
import random

with open('nspst', 'rb') as f:
    ztl = pickle.load(f)
ztl = [(z[0], z[1:5]) for z in ztl]
ztl = dict(ztl)

with open('nan', 'rb') as f:
    nan = pickle.load(f)

id_set = set()


data_dir = 'png_data/png_v0'
for d1 in os.listdir(data_dir):
    print(d1)
    for d2 in os.listdir(data_dir + '/' + d1):
        for d3 in os.listdir(data_dir + '/' + d1 + '/' + d2):
            for file in os.listdir(data_dir + '/' + d1 + '/' + d2 + '/' + d3):
                id_set.add(int(file.split('.')[0]))

print(len(id_set))

proxys = '117.43.28.21:29724'
def get_dict_from_table(trs):

    table_dict = {}

    for tr in trs:

        if len(tr.find_all('td')) == 2:

            k, v = [t.text for t in tr.find_all('td')]

            k = unidecode(k.replace(':', '').strip())

            v = unidecode(v.replace(':', '').strip())

            table_dict[k] = v

    return table_dict

def get_metadata(metadata_html):
    """dictionary of all metadata

    Parameters

    ----------

    metadata_html: html

        html data containing metadata



    Returns

    -------

    dic: dictionary

        dictionary of all data



    """

    soup = BeautifulSoup(metadata_html, 'html.parser')

    table1 = soup.find_all('tbody')[1]

    trs = table1.find_all('tr')[2].find_all('tr')

    d1 = get_dict_from_table(trs)

    table2 = soup.find_all('tbody')[4]

    trs = table2.find_all('tr')

    d2 = get_dict_from_table(trs)

    table3 = soup.find_all('tbody')[8]

    trs = table3.find_all('tr')

    d3 = get_dict_from_table(trs)

    dic = {}

    dic.update(d1)

    dic.update(d2)

    dic.update(d3)

    return dic

def download_png(nmo):
    global global_q

    neuron_id = nmo + 1
    if neuron_id in id_set:
        return

    proxy = random.choice(proxys)
    proxy_support = urllib.request.ProxyHandler({'http': proxy})
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent',
                          'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)')]
    urllib.request.install_opener(opener)


    cell_type = [0, 0, 0]
    cell_type[2] = ztl[neuron_id][1]
    cell_type[1] = ztl[neuron_id][2]
    cell_type[0] = ztl[neuron_id][3]
    for i in range(3):
        cell_type[i] = cell_type[i].replace('/', ' or ')

    d1 = data_dir + '/' + cell_type[2]
    d2 = data_dir + '/' + cell_type[2] + '/' + cell_type[1]
    d3 = data_dir + '/' + cell_type[2] + '/' + cell_type[1] + '/' + cell_type[0]
    if cell_type[2] not in os.listdir(data_dir):
        os.mkdir(d1)
    if cell_type[1] not in os.listdir(d1):
        os.mkdir(d2)
    if cell_type[0] not in os.listdir(d2):
        os.mkdir(d3)

    png_url = 'http://neuromorpho.org/images/imageFiles/' + nan[neuron_id][0] + '/' + nan[neuron_id][1] + '.png'
    # print(png_url)

    try:
        urllib.request.urlretrieve(png_url, d3 + '/' + str(neuron_id) + '.png')
        print(neuron_id)
    except urllib.error.HTTPError as e:
        print(e)





if __name__ == '__main__':
    global_q = Queue()
    t_pool = threadpool.ThreadPool(200)  # 线程池

    # 任务参数列表
    arg_list = [nmo for nmo in range(0, 110000)] #107395
    random.shuffle(arg_list)
    requests = threadpool.makeRequests(download_png, arg_list)

    # 把任务放入线程池
    [t_pool.putRequest(req) for req in requests]
    # 等待所有任务处理完成，则返回，如果没有处理完，则一直阻塞
    t_pool.wait()
    t_pool.dismissWorkers(5, do_join=True)


