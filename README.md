# neuromorpho_neuron_type_classification

Use RNN to classify unstructured swc data.

(1)The goal of the mission is to classify 102,534 neurons according to their neuronal types. The first step is to divide the neurons into 'interneuron', 'principal cell', 'Glia', 'Not reported', 'sensory receptor', etc. Class, considering that the data set is mainly 'interneuron' and 'principal cell',so it can be approximated as a two-class problem.

(2)The data consists of 102,534 swc files, each of which is a tree structure representation of a neuron.The data structure is as follows：
```sh
 0 [0.0, 0.0, 0.0] 1 -1 7.64 0 0.0 0.0
 1 [6.54, 3.93, 0.0] 1 0 7.64 -1 7.63 0.0
 2 [-6.54, -3.93, 0.0] 1 0 7.64 -1 7.63 0.0
 3 [-4.89, 11.54, -0.27] 4 0 1.09 1 12.54 71.13
 4 [2.08, -13.49, 6.19] 3 0 0.54 1 14.99 66.08
 ……
 ```
The specific representation is: each swc file consists of several rows, each row representing a node on a neuron, which consists of the following items:
```sh
 ['id', 'P', 'type', 'parent', 'width', 'branch_level', 'path_length', 'degree'], where
Id: node number
P: node coordinates, with the cell body as the origin, is a (x, y, z) triplet
type: 1-core, 2-axon, 3-the end of the dendritic, 4-apical dendrite
parent: parent node number
width: diameter of the neuron
branch_level: node level, -1-end point
path_length: node to cell distance
degree: the angle between two child nodes
 ```
(3)For train and validate model:
```sh
python RECTREE_train.py
```
Or modify the dataset_dir parameter in RECTREE_train.py is also ok.

(4)pip install -r requirements.txt in a virtual env functions the same as a yml.

additional packages for download neorumorpho data.
```python
numpy
pandas
urllib
ast
re
json
urllib2
bs4
unidecode
```
(5)The RNN is used to classify the neurons. Each swc file is regarded as a sequence of [node number * 10], and in front of a double-layer RNN is a CNN layer. The RNN layer through the fully connected layer outputs the classification result. The accuracy is 65%.

The results are shown in the figure below.

![Image text](https://github.com/thomasaimondy/neuromorpho_neuron_type_classification/blob/master/images/result.jpg)
