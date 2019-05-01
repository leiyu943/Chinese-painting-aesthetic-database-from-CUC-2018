# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 22:55:47 2019
@author: leiyu94

小论文的完整程序,分类的步骤为：
1.使用RFECV和SVM进行递归式特征筛选；
2.将筛选出的特征用于极度随机树、SVM、随机森林、Adaboost等特征分类；
3.返回预测精度与训练集测试精度；

似乎在十折交叉验证的时候把数据集分割的时候用Stratify一下，极度随机树的精度就会蹭蹭蹭上去，高于0.8.
希望是真的_(:з」∠)，跑起来也是真的慢
"""
from __future__ import division  
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm,preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#读取文件
path='D:/feature/2018-12-23_20_39_40_120_features_with_preprocdessing.csv'
data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=0)
train_data = data.values
train_label =np.array(data.index)

#特征归一化
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)

#特征筛选
print("特征筛选已开始")
clf = svm.SVC(kernel='linear')
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(8),
              scoring='accuracy')
rfecv.fit(train_data, train_label)
print("最佳特征数目为 : %d" % rfecv.n_features_)
x_label = range(1, len(rfecv.grid_scores_) + 1)
y_label = rfecv.grid_scores_
support=rfecv.support_
plt.figure()
plt.xlabel(u"所选特征数量")
plt.ylabel(u"交叉验证得分（分类精度）")
plt.plot(x_label, y_label)
plt.show()

#获取有效特征
train_data1=rfecv.transform(train_data)

#准备需要验证的分类器
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
classifiers = {
    'logistic re ': LogisticRegression(C=1.1, penalty='l1', tol=0.01),#
    'SVC        ': SVC(kernel="linear"),
    'MLP         ': MLPClassifier(hidden_layer_sizes=(300,4),solver ='adam',
                                  activation ='tanh',alpha=1e-5,learning_rate='invscaling',
                                  learning_rate_init=1e-3,power_t=0.5,
                                  max_iter=200,shuffle =False,random_state=42,
                                  tol=1e-5,early_stopping =False,beta_1=0.9999999,
                                  beta_2=0.99999,epsilon=1e-9),
    'KNN         ': KNeighborsClassifier(n_neighbors =20,weights='distance',
                                         algorithm ='auto',leaf_size =50,
                                         p=1,n_jobs =1),      
    'RandomForest': RandomForestClassifier(n_estimators=150, max_depth=10,  max_features=1,
                                           class_weight="balanced_subsample", ),  # clf.feature_importances_
    'ExtraTrees  ': ExtraTreesClassifier(n_estimators=180, max_depth=30,min_samples_split=20,
                                         min_samples_leaf =1,min_weight_fraction_leaf=0,
                                         max_features =None,max_leaf_nodes=None,n_jobs=3 ,
                                         warm_start =True,
                                         bootstrap=None),  #dual=True
#这个Adaboost会报错，暂时注释掉了。                                         
#    'AdaBoost1  ': AdaBoostClassifier(LogisticRegression(C=100,penalty='l1',tol=0.0002,
#                                                          fit_intercept =False,intercept_scaling =1,
#                                                          max_iter=9,multi_class='ovr',verbose =1.5,
#                                                          warm_start=True,n_jobs=0),
#                                      n_estimators=5, learning_rate=0.1109075,algorithm='SAMME.R'),#适应性提升
    'AdaBoost2  ': AdaBoostClassifier(SVC(kernel="linear",	C=3,degree=3,shrinking =False,
                                          probability=True,tol =1e-4,
                                          verbose =False,decision_function_shape ='ovr'),
                                      n_estimators=5, learning_rate=0.1109075,algorithm='SAMME'),#适应性提升
    
    'Gradient Boo': GradientBoostingClassifier(n_estimators=300, learning_rate=1,
                                               loss ='deviance',subsample =1,criterion='friedman_mse',
                                               min_samples_split=0.3,min_impurity_decrease=1e-7,
                                               max_depth=4, random_state=0), #梯度提升数4
    'LinearDisc  ': LinearDiscriminantAnalysis(solver='lsqr',shrinkage ='auto',
                                               n_components=4,store_covariance=False,
                                               tol=1e-4),#线性判别分析          
    'MultinomialNB': MultinomialNB(alpha =635,fit_prior=True),
    'GaussianNB': GaussianNB(),
#    'SGDClassifier': SGDClassifier(loss= 'hinge', penalty='l1', alpha=1e-4, 
#                                   random_state=42, max_iter=8, tol=0.5),
    
    'bagging1'      : BaggingClassifier(MLPClassifier(hidden_layer_sizes=(90, ),
                                                     activation ='tanh',alpha=0.0005),
                                       max_samples=0.5, max_features=0.5) ,
    'bagging2'      : BaggingClassifier(LogisticRegression(C=1.1,penalty='l1', tol=0.01),
                                       max_samples=0.5, max_features=0.5) ,
    'bagging3'      : BaggingClassifier(SVC(kernel="linear", C=0.02,decision_function_shape ='ovo'),
                                       max_samples=0.5, max_features=0.5)     
    }


print '开始验证'
cv = StratifiedKFold(n_splits=10)
i = 0
X=train_data1
y=train_label

#from __future__ import division  
for name, clf in classifiers.items():
#    classifier = clf
    scores=[]
    for train, test in cv.split(X, y):
        classifier = clf
        if name=='ExtraTrees  ':
            classifier.set_params(n_estimators=180, max_depth=30,min_samples_split=20,   
                                  min_samples_leaf =1,min_weight_fraction_leaf=0,  
                                  max_features =None,max_leaf_nodes=None,n_jobs=3 ,  
                                  warm_start =True,    bootstrap=None) 
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        
        classifier.fit(X_train, y_train)
        '''
        注意：以下四种方式只能单次运行一种
        '''
        #训练集测试
#        train_pred = clf.predict(X_train)        
#        train_acc_sum = (train_pred == y_train).sum()
        
        #训练集每类准确率
#        train_pred = clf.predict(X_train)   
#        train_acc=[]
#        for kk in range(5):
#            train_acc.append(sum((train_pred == y_train)*(y_train == 1+kk))/sum (y_train==1+kk))                          
#        scores.append(train_acc)
    
        #测试集测试
#        train_pred = clf.predict(X_test)   
#        train_acc_sum = (train_pred == y_test).sum()    
        
        #求每个美感类的测试集查准率
        train_pred = classifier.predict(X_test)  
        train_acc=[]
        for kk in range(5):
            if not sum (train_pred==1+kk)==0:
                train_acc.append(sum((train_pred == y_test)*(train_pred== 1+kk))/sum (train_pred==1+kk))
            else:                    
                train_acc.append(0)
                
        scores.append(train_acc)
        
        #求每个美感类的测试集查全率
#        train_pred = clf.predict(X_test)  
#        train_acc=[]
#        for kk in range(5):
#            if not sum (train_pred==1+kk)==0:
#                train_acc.append(sum((train_pred == y_test)*(y_test== 1+kk))/sum (y_test==1+kk))
#            else:                    
#                train_acc.append(0)
#                
#        scores.append(train_acc)
        
        
        
    scores = [[row[i] for row in scores] for i in range(len(scores[0]))]   
        
        #求预测精度
#        train_acc = train_acc_sum/len(train_pred)       
#        scores.append(train_acc)
##    scores = cross_val_score(clf, train_data, train_label,cv=10)
#    scores = cross_val_score(clf, train_data1, train_label,cv=StratifiedKFold(10))
##    KFold(n_splits=2, random_state=None, shuffle=False)
    print(name,'\t--> ',np.mean(scores,axis=1))
