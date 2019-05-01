# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:57:41 2018

@author: xielingyun
"""
from __future__ import division  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold

path = 'G:\\2018hanjia\\feature\\2018-12-23_20_39_40_141_features_with_preprocdessing - ml.csv'
data = pd.read_csv(path, dtype = np.float64,  converters={'class':int},index_col=[0,1,2,3,4])
train_data = data.values
class_names = np.array(data.index.names)
train_multilabel = np.zeros((len(train_data),len(class_names)))
total_number = len(train_data)*len(class_names)

#数据预处理，压缩全体数据的动态范围
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)

#读取多标签到二维矩阵
count = 0
for s in class_names:
    train_multilabel[:,count] = data.index.get_level_values(s)
    count = count + 1

#clf = OneVsRestClassifier(SVC(kernel='linear'))
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))

#交叉检验开关：设为1则进行随机拆分的交叉检验，设为其他值则进行集内测试和评估
crossval = 1 
if crossval == 1:
    train_loss_rate = 0
    train_falsepositive_rate = 0
    test_loss_rate = 0
    test_falsepositive_rate = 0
    train_acc = 0
    test_acc = 0
    kfold = 4
    kf = KFold(n_splits=kfold, shuffle=True)
    for train, test in kf.split(train_data):
        x = train_data[train]
        y = train_multilabel[train]
        tx = train_data[test]
        ty = train_multilabel[test]
        clf.fit(x,y)
        
        pred = clf.predict(x)
        train_result = y - pred
        train_number = len(x) * len(class_names)
        train_acc = train_acc + (train_result == 0).sum()/train_number
        train_loss_rate = train_loss_rate + (train_result == 1).sum()/train_number
        train_falsepositive_rate = train_falsepositive_rate + (train_result == -1).sum()/train_number
        
        pred = clf.predict(tx)
        test_result = ty - pred
        test_number = len(tx) * len(class_names)
        test_acc = test_acc + (test_result == 0).sum()/test_number
        test_loss_rate = test_loss_rate + (test_result == 1).sum()/test_number
        test_falsepositive_rate = test_falsepositive_rate + (test_result == -1).sum()/test_number
        
    train_acc = train_acc/kfold
    train_loss_rate = train_loss_rate/kfold
    train_falsepositive_rate = train_falsepositive_rate/kfold
    test_acc = test_acc/kfold
    test_loss_rate = test_loss_rate/kfold
    test_falsepositive_rate = test_falsepositive_rate/kfold
    
    print("训练集的总体分类正确率: %f%%" %(train_acc*100))
    print("训练集的漏检率: %f%%" %(train_loss_rate*100))
    print("训练集的误识率: %f%%" %(train_falsepositive_rate*100))
    print("测试集的总体分类正确率: %f%%" %(test_acc*100))
    print("测试集的漏检率: %f%%" %(test_loss_rate*100))
    print("测试集的误识率: %f%%" %(test_falsepositive_rate*100))
    print("说明：这里三个指标的分母都是样本数和类别数的乘积")
    
else:
    clf.fit(train_data, train_multilabel)
    pred = clf.predict(train_data)
    result = train_multilabel - pred
    train_acc = (result == 0).sum()/total_number
    loss_rate = (result == 1).sum()/total_number
    false_positive_rate = (result == -1).sum()/total_number
    print("训练集的总体分类正确率: %f%%" %(train_acc*100))
    print("训练集的漏检率: %f%%" %(loss_rate*100))
    print("训练集的误识率: %f%%" %(false_positive_rate*100))
    print("说明：这里三个指标的分母都是样本数和类别数的乘积")

