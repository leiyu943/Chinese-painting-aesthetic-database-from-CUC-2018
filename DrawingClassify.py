# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:13:47 2018

@author: Xie Lingyun
"""
from __future__ import division  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm, tree
from sklearn.model_selection import train_test_split

path = 'D:/feature/feature0720-5lei.csv'
#此处是调用pandas的读取csv格式的函数，中间设定了数据格式，同时把
#标签那一列读为int类型，设定标签列为第90列
data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=90)
mydata = data.values
mylabel = np.array(data.index)

#随机将数据分为训练集和测试集，测试集的比例可以通过test_size设定
train_data, test_data, train_label, test_label = train_test_split(
        mydata, mylabel, test_size=0.25, random_state=0)
#bad = np.where(np.isnan(pp)) #做数据检查
#print(bad)

clf = svm.SVC()
#clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_label)

train_pred = clf.predict(train_data)
train_acc_sum = (train_pred == train_label).sum()
train_acc = train_acc_sum/len(train_pred)
print("Train Set Result: %f%%" %(train_acc*100))
test_pred = clf.predict(test_data)
test_acc_sum = (test_pred == test_label).sum()
test_acc = test_acc_sum/len(test_pred)
print("Test Set Result: %f%%" %(test_acc*100))
