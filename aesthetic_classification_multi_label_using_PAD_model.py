# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:00:54 2019

@author: Administrator
"""
from __future__ import division  

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm,preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
path='G:/2018hanjia/feature/2018-12-23_20_39_40_120_features_with_PAD&multi_label -withoutfilename.csv'
data = pd.read_csv(path,dtype = np.float64, index_col=range(9))
train_data = data.values
train_label =np.array(data.index.names)
count = 0
labels = np.zeros((len(train_data),len(train_label)))
for s in train_label:
    labels[:,count] = data.index.get_level_values(s)
    count = count+1
PADv=labels[:,[0,1,2]]
def int1(x):
    return int(x)
multilabels=[]
count=0
for line in labels:
    multilabels.append(map(int,line[3:8]) )
    count=count+1
singlelabels=map(int,labels[:,8])

#特征归一化
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)
#特征筛选
clf = svm.SVC(kernel='linear')
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(10),
              scoring='accuracy')
rfecv.fit(train_data, singlelabels)
print '筛选结束'
#有效特征标签
support=rfecv.support_
#获取有效特征数据
train_data=rfecv.transform(train_data)

from sklearn import svm,preprocessing,linear_model,neighbors,tree,dummy,cross_decomposition
from sklearn import ensemble
from sklearn import gaussian_process,kernel_ridge,isotonic
from sklearn.ensemble import ExtraTreesClassifier
Regressors={
#        'pls':cross_decomposition.PLSRegression(),报错
        'gradient boosting':ensemble.GradientBoostingRegressor(),
#        'gaussian':gaussian_process.GaussianProcessRegressor(),报错
#        'isotonic':isotonic.IsotonicRegression(),报错
        'kernelridge':kernel_ridge.KernelRidge(),
        'ARD':linear_model.ARDRegression(),
        'bayesianridge':linear_model.BayesianRidge(),
#        'elasticnet':linear_model.ElasticNet(),#报错
        'HuberRegressor':linear_model.HuberRegressor(),
        'LinearRegression':linear_model.LinearRegression(),
#        'logistic':linear_model.LogisticRegression(),报错
#        'linear_model.RidgeClassifier':linear_model.RidgeClassifier(),报错
        'k-neighbor':neighbors.KNeighborsRegressor(),
        'SVR':svm.LinearSVR(),
        'NUSVR':svm.NuSVR(),
        'extra tree':tree.ExtraTreeRegressor(),
        'decesion tree':tree.DecisionTreeRegressor(),
#        'random losgistic':linear_model.RandomizedLogisticRegression(),报错
#        'dummy':dummy.DummyRegressor()报错
        }

#回归分析
from sklearn.metrics import mean_squared_log_error
cv = StratifiedKFold(n_splits=5)
i = 0
X=train_data
y=PADv
z=np.array(multilabels)
#clf=ExtraTreesClassifier()
zong=[]#回归部分结果
zong1=[]
#switch表示仅PAD用于分类，1表示PAD与原有特征共同用于分类,2为仅特征
switch=1


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
clf = OneVsRestClassifier(ExtraTreesClassifier())
for name,reg in Regressors.items():
#    test_falsepositive_rate = 0
#    print name+':'
    test_acc=[]
    test_loss_rate =[]
    test_falsepositive_rate=[]
    for train, test in cv.split(X, singlelabels):  
        test_results=[]
        test_realvalue=[]
        X_train, X_test = X[train], X[test]        
        z_train, z_test = z[train], z[test]     
        MSE=[]
        RMSE =[]
        MAE=[]
        MSLE=[]
        RMSLE =[]
        for i in range(3):
            reg.fit(X_train,PADv[train,i])
            pre=reg.predict(X_test)
            test_results.append(pre)
            test_realvalue.append(np.transpose(PADv[test,i]))
            MSE.append(mean_squared_error(pre, np.transpose(PADv[test,i]))**0.5)
#            RMSE.append(mean_squared_error(pre, np.transpose(PADv[test,i])))
            MAE.append(mean_absolute_error(pre, np.transpose(PADv[test,i])))
            MSLE.append(mean_squared_log_error(pre, np.transpose(PADv[test,i])))
        test_results=np.transpose(test_results)
        RMSLE =np.array( MSLE) ** 0.5
        RMSE = np.array(MSE) ** 0.5
        zong.append([name,MSE,RMSE,MAE,MSLE,RMSLE])
#        print [name,MSE,RMSE,MAE,MSLE,RMSLE]
        if switch==0:
            clf.fit(PADv[train,:],z_train)
#            clf_result=clf.predict(PADv[test,:])   #原PAD    
            clf_result=clf.predict(test_results)#预测PAD
        elif switch==1:  
            traindata=list(np.transpose(PADv[train,:]))+list(np.transpose(X_train))  
            traindata=np.transpose(traindata)      
#            testdata=list(np.transpose(PADv[test,:]))+list(np.transpose(X_test))   #原PAD
            testdata=list(np.transpose(test_results))+list(np.transpose(X_test))#预测PAD
            testdata=np.transpose(testdata)     
            clf.fit(traindata,z_train)     
            clf_result=clf.predict(testdata)
        else:
            traindata=X_train
            testdata=X_test
            clf.fit(traindata,z_train)
            clf_result=clf.predict(testdata)
        test_result =z_test - clf_result         
        test_number = len(test_result) * 5
        test_acc.append((test_result == 0).sum()/test_number)
        test_loss_rate.append((test_result == 1).sum()/test_number)
        test_falsepositive_rate.append((test_result == -1).sum()/test_number)
        #顺序为：'正确率、漏检率、误识率: '
    print name,np.mean(test_acc),np.mean(test_loss_rate),np.mean(test_falsepositive_rate)
    
#
#import csv
#fileHeader = ["regressor", "result",'real value']
#csvFile = open("PAD_regressor_results.csv", "w")
#writer = csv.writer(csvFile)
#writer.writerow(fileHeader)
#writer.writerow(zong)
##writer.writerow()
#csvFile.close()