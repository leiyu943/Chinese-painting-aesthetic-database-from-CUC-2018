# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 15:58:25 2018

@author: cal
"""


from __future__ import division  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold 
from sklearn import feature_selection
path = 'E:/feature/2018-12-23_20_39_40_141_features_with_preprocdessing - ml.csv'
data = pd.read_csv(path, dtype = np.float64, index_col=[0,1,2,3,4])
train_data = data.values
class_names = np.array(data.index.names)
labels = np.zeros((len(train_data),len(class_names)))
count = 0
for s in class_names:
    labels[:,count] = data.index.get_level_values(s)
    count = count+1
train_label = np.ones(len(train_data))
for j in np.arange(len(train_data)):
    temp = np.argwhere(labels[j,:]==max(labels[j,:]))
    train_label[j] = temp[0]


#数据预处理，压缩全体数据的动态范围
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=80)
train_data = fs.fit_transform(train_data, train_label)
qw=fs.get_support()


results = np.zeros((len(train_data),len(class_names)))
lc = len(class_names)
train_cm = np.zeros([lc,lc],dtype=int)
test_cm = np.zeros([lc,lc],dtype=int)
test_acc = 0
train_acc = 0

#from sklearn import svm
#clf = svm.SVR()
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
clf = KNeighborsRegressor(n_neighbors=20)
#clf =ExtraTreesRegressor(n_estimators=180, max_depth=30,min_samples_split=20,   
#                         min_samples_leaf =1,min_weight_fraction_leaf=0,   
#                        n_jobs=3, )

kfold = 5
skf = StratifiedKFold(n_splits=kfold, shuffle=True)
for train, test in skf.split(train_data,train_label):
    x = train_data[train]
    y = labels[train]
    ym = train_label[train]
    tx = train_data[test]
    ty = labels[test]
    tym = train_label[test]
    
    train_results = np.zeros((len(ym),lc))
    test_results = np.zeros((len(tym),lc))
    for i in np.arange(lc):
        yl = y[:,i]
        clf.fit(x,yl)
        train_results[:,i] = clf.predict(x)
        test_results[:,i] = clf.predict(tx)
           
    max_train_results = np.zeros(len(ym))
    for j in np.arange(len(y)):
        train_results[j,:] = train_results[j,:]/sum(train_results[j,:])
        temp = np.argwhere(train_results[j,:]==max(train_results[j,:]))
        max_train_results[j] = temp[0]
        
    max_test_results = np.zeros(len(tym))
    for k in np.arange(len(ty)):
        test_results[k,:] = test_results[k,:]/sum(test_results[k,:])
        temp = np.argwhere(test_results[k,:]==max(test_results[k,:]))
        max_test_results[k] = temp[0]
    
    train_acc = train_acc + (max_train_results == ym).sum()/len(ym)
    train_cm = train_cm + confusion_matrix(ym, max_train_results)
    test_acc = test_acc + (max_test_results == tym).sum()/len(tym)
    test_cm = test_cm + confusion_matrix(tym, max_test_results)
train_acc = train_acc/kfold
test_acc = test_acc/kfold

#结果输出
print("训练集的分类正确率: %f%%" %(100*train_acc))
print('训练集的混淆矩阵：')
print(train_cm)
print("\n测试集的分类正确率: %f%%" %(100*test_acc))
print('测试集的混淆矩阵：')
print(test_cm)
