# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:06:26 2019
采用特征预测PAD值
@author: leiyu94
"""


from __future__ import division  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm,preprocessing,linear_model,neighbors,tree,dummy,cross_decomposition
from sklearn import ensemble

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

#读取文件
path='D:\\feature\\2018-12-23_20_39_40_120_features_with_PAD_reg.csv'
data = pd.read_csv(path,dtype = np.float64, index_col=[0,1,2])
train_data = data.values
train_label =np.array(data.index.names)
labels = np.zeros((len(train_data),len(train_label)))
count = 0
for s in train_label:
    labels[:,count] = data.index.get_level_values(s)
    count = count+1


#读取情感标签
emot=[]
for num in labels[:,2]:
    emot.append(int(num))
emot=np.array(emot)

#数据预处理，压缩全体数据的动态范围
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)

#特征筛选-使用递归+svm
#print("特征筛选已开始")
#clf = svm.SVC(kernel='linear')
#rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(5),
#              scoring='accuracy')
#rfecv.fit(train_data, emot)
##x_label = range(1, len(rfecv.grid_scores_) + 1)
##y_label = rfecv.grid_scores_
##有效特征标签
#support=rfecv.support_
##获取有效特征数据
#train_data=rfecv.transform(train_data)


#特征筛选，使用RLR
#from sklearn.linear_model import RandomizedLogisticRegression as RLR
#rlr=RLR() 
#rlr.fit(train_data,probs)
#rlr.get_support() 

#准备回归分类器
import sklearn
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

cv = StratifiedKFold(n_splits=5)
i = 0
X=train_data
#y=probs
#z=labels[:,2]
y=np.array(range(len(X)))

#y=int("".join(list(map(str,))))
#from functools import reduce
#y=reduce(lambda x,y: 10*x+y, (y))
emotion=['P','A','D']
for ii in range(3):
    z=labels[:,ii]
    zz=(z>5)
    for name, rgs in Regressors.items():
        regressor = rgs
        scores=[] 
        for train, test in cv.split(X,zz):  
            X_train, X_test = X[train], X[test]
            #        y_train, y_test = y[train], y[test]     
            z_train, z_test = z[train], z[test] 
            train_results=[]
            test_results=[]
        
        #建立回归模型，获得回归分析
            rgs.fit(X_train, z_train)            
            train_results.append(rgs.predict(X_train))
            test_results.append(rgs.predict( X_test))        
            z_result=np.array(np.transpose(test_results))    
            PP=np.sum((np.transpose(z_result<=5)==(z_test<=5))*(z_test<=5))/np.sum(z_test<=5)
            RR=np.sum((np.transpose(z_result<=5)==(z_test<=5))*(z_test<=5))/np.sum(z_result<=5)
#            FP=(z_result>5)==(z_test>5)
            
            scores.append(PP)
        print name+'_'+emotion[ii]+':'+str(scores)
        #训练组训练
#            clf.fit(y_train,z_train)
        #测试组测试        
#            classes=clf.predict(z_result)
        
        #查全率和查准率
          
#            score=[]
#        for xx in range(2):
#            R=sum((classes==xx)*(z_test==xx))/sum(z_test==xx)
#            P=sum((classes==xx)*(z_test==xx))/sum(classes==xx)
#            score.append(R)
#            score.append(P)
#        scores.append(score)
