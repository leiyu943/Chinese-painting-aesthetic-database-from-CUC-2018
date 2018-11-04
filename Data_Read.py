# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 21:36:36 2018
本示例给出用于读取csv文件和arff文件的代码，特征数据保存到train_data，标签保存到train_label
其中，读取arff文件有两个方法，一个是调用scipy类库里面io模块的loadarff函数；另一个是采取直接逐行读取。
读取csv文件则调用了Pandas类库的read_csv函数。
@author: Xie Lingyun
"""

import operator
import numpy as np
import pandas as pd
from scipy.io import arff

path = 'E:/Data/music_emotion.csv'  #所有方法都用相同的文件

#读取arff文件的第1个方法：
data, meta = arff.loadarff(path)
sample_num = len(data)
feature_num = len(data[0])-1 #因为arff文件数据部分的最后一维是标签，所以计算特征数目时要减去
tmpdata = []
for n in range(len(data)):
    tmpdata.append(list(data[n]))  #loadarff读出来的数据格式不是list，为了应用方便，先在此处转为list格式    
train_data = []
train_label = np.ones(sample_num, dtype=np.int16)
for i in range(sample_num):
    train_data.append(tmpdata[i][0:feature_num])
    train_label[i] = tmpdata[i][feature_num]


#读取arff文件的第2个方法：
f = open(path)
lines = f.readlines()
count = 0
train_data = []
train_label = []
for l in lines:
    if count == 1:
        content = l.split(',')
        tmp = len(content)-1
        tmplabel = content[tmp].strip('\n')
        tmplabel = int(tmplabel)
        if tmplabel == 5:
            tmplabel = 0
        train_label.append(tmplabel)
        content.remove(content[tmp])
        content = list(map(lambda x: float(x), content))
        train_data.append(content)
    if operator.eq(l,'@data\n') == True:
        count = 1  


#读取csv文件的方法：
data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=0)
train_data = data.values
train_label = np.array(data.index)