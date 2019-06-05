import urllib
import ast
import numpy as np
import pandas as pd
import re
import json
from urllib2 import Request, urlopen
from bs4 import BeautifulSoup
from unidecode import unidecode
import pickle

np.set_printoptions(threshold=np.inf)

__all__ = [
    'get_dict_from_table',
    'find_archive_link',
    'get_metadata',
    'swc',
    'download_neuromorpho',
]
def get_dict_from_table(trs):
    table_dict = {}
    for tr in trs:
        if len(tr.find_all('td')) == 2:
            k, v = [t.text for t in tr.find_all('td')]
            k = unidecode(k.replace(':', '').strip())
            v = unidecode(v.replace(':', '').strip())
            table_dict[k] = v
    return table_dict


def find_archive_link(a):
    for a in soup.find_all('a'):
        if a is not None and a.find('input') is not None:
            if 'Link to original archive' == a.find('input').attrs['value']:
                archive_link = a.attrs['href']
    return a

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

def swc(swc_html):
    """
    Getting swc from html file
    """
    neuron = np.zeros([1,7])
    for line in swc_html:
        line = line.lstrip()
        if line[0] is not '#' and len(line) > 2:
            l= '['+re.sub('(\d) +', r'\1,', line[:-1])+']'
            l= re.sub('(\.) ', r'\1,', l)
            neuron = np.append(neuron, np.expand_dims(np.array(ast.literal_eval(l)),axis=0),axis=0)
    return neuron[1:,:]

def download_neuromorpho(start_nmo=1, end_nmo=50, show_progress=False):
    """Downloading neuron morphologies with their metadata from neuronmorpho.org
    Parameters
    ----------
    start_nmo: int
        starting index of neuron
    end_nmo: int
        last index of neuron
    show_progress: boolean
        to show the indexing during downloads
    
    Returns
    -------
    all_neuron: Dataframe
        Dataframe containing all the metadata and swc matrix of morphologies
    errors: numpy
        the nmo index of neurons that could not be downloaded (got an error). 
    """
    all_neuron = []
    errors = []
    for nmo in range(start_nmo, end_nmo,1):
        try:
            if show_progress:
                print(nmo)
            txt = urllib.urlopen("http://neuromorpho.org/api/neuron/id/"+str(nmo)).read()
            neuron_dict = json.loads(txt.decode("utf-8"))
            if len(neuron_dict) != 0:
#                 neuron_dict = neuron_dict[0]
                neuron_name = neuron_dict['neuron_name']
                #print(neuron_name)
                archive_name = re.sub(' ', '%20',neuron_dict['archive'])
                neuromorpho_link = 'http://neuromorpho.org/neuron_info.jsp?neuron_name='+ neuron_name
#         http://neuromorpho.org/dableFiles/wearne_hof/CNG%20version/cnic_001.CNG.swc
                neuromorpho_link_swc = 'http://neuromorpho.org/dableFiles/' + archive_name.lower() + '/CNG%20version/' + neuron_name + '.CNG.swc'
#                 print(neuromorpho_link)
#                 print(neuromorpho_link_swc)
                metadata_html = urllib.urlopen(neuromorpho_link).read()
#                 print(metadata_html)
                response = urllib.urlopen(neuromorpho_link_swc)
                if response.getcode() != 404:
                    swc_html = response.readlines()
                    metadata_html = metadata_html.replace('</tr>\n</tr>','</tr>',1)
                    dic_all = get_metadata(metadata_html)
                    dic_all['link in neuromorpho'] = str(neuromorpho_link)
                    dic_all['swc'] = swc(swc_html)
#                     print(dic_all)
                    all_neuron.append(dic_all)
#                     print(all_neuron)
        except:
            errors.append(nmo)
            print('excepterror_'+str(nmo))
    #all_neuron = pd.DataFrame(all_neuron)
    
    return all_neuron