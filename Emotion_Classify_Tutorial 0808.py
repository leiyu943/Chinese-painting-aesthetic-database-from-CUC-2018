# -*- coding: utf-8 -*-
"""
这是机器学习的入门实例。以音乐情感的带分类特征数据文件为对象，涉及到（1）数据的读取，
（2）数据预处理，(3)特征选择，（4）训练分类器，（5）结果评估 等五个步骤。
需预先安装scikit-learn

@author: Xie Lingyun
"""
import operator
#from sklearn import svm
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn import feature_selection
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
#from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# meta-estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import BaggingClassifier

import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
 #sklearn.ensemble.BaggingClassifier

import numpy as np
import scipy.io as scio
import csv

#dataFile = 'D:/feature/sift_knn_100.mat'
#data = scio.loadmat(dataFile)
#sift=data['biao']

#步骤1：读取arff文件的数据，该民乐情感数据集共有207个样本，192个特征
#       有四类情感标签：1-anger(55个),2-depress(68个),3-happy(41个),4-easy(43个)
path='D:/feature/2018-12-23_20_39_40_142_features_with_preprocdessing.csv'
#my_matrix = np.loadtxt(open(path,"rb"),delimiter=",",skiprows=0)


data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=0)
train_data = data.values
train_label =np.array(data.index)

#f = open('D:/feature/feature_with_preproccessing.csv')

#data = csv.reader(open(path))



#lines = f.readlines()
#count = 0
#train_data = []
#train_label = []
#
#
#with open(path, 'rb') as csvfile:
#    train_data = csv.reader(csvfile, delimiter=',', quotechar='|')
#
#for l in data:
#    if count == 1:
#        content = l.split(',')#获得该行数据
#        tmp = len(content)-1#获得数据长度
#        tmplabel = content[tmp].strip('\n') #获取最末尾的标签 并移除数据末尾的'\n'
#        tmplabel = int(tmplabel) #将标签转化为
#        if tmplabel == 6:
#            tmplabel = 0
#        train_label.append(tmplabel)
#        content.remove(content[tmp])
#        content = list(map(lambda x: float(x), content))
#        train_data.append(content)
#    if operator.eq(l,'@data\n') == True:
#        count = 1  

#for l in lines:
#    content = l.split(',')
#    tmp = len(content)-1
#    tmplabel = content[tmp].strip('\n')
#    if tmplabel == 5:
#        tmplabel = int(tmplabel)
#        tmplabel = 0
#        train_label.append(tmplabel)
#        content.remove(content[tmp])
#        content = list(map(lambda x: float(x), content))
#        train_data.append(content)
##    if operator.eq(l,'@data\n') == True:
##        count = 1  
#train_data .fillna('0')
#data=csv.reader(open('D:/feature/feature0720-5lei.csv','rb'))
#data.readlines
#步骤2：特征预处理，此处将特征数据范围变换为[0,8]区间
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,5))
train_data = min_max_scaler.fit_transform(train_data)

#步骤3：特征选择，做ANOVA分析，以F值高低排序，选择前K个特征
#X_train_fs = SelectKBest(k=3).fit_transform(train_data, train_label)
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=23.94)
X_train_fs = fs.fit_transform(train_data, train_label)
qw=fs.get_support()
#拆分操作：随机将数据拆分为训练集和测试集，测试集的比例可以通过test_size设定
#train_data1, test_data, train_label1, test_label = train_test_split(
#        train_data, train_label, test_size=0.2, random_state=0,stratify=train_label)

#步骤4：训练分类器，此处用svm分类器为例
#clf=AdaBoostClassifier(n_estimators=100)#35.7,25
#clf = DecisionTreeClassifier()#max_depth=5,99.73,30or26or25
#clf = RandomForestClassifier(n_estimators=10)#95,30;92;32;94,33
#clf = svm.SVC(kernel='linear')#51,30;51,30
#clf=KNeighborsClassifier(11)#35,8,38;11,40,37
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
    'AdaBoost1  ': AdaBoostClassifier(LogisticRegression(C=100,penalty='l1',tol=0.0002,
                                                          fit_intercept =False,intercept_scaling =1,
                                                          max_iter=9,multi_class='ovr',verbose =1.5,
                                                          warm_start=True,n_jobs=0),
                                      n_estimators=5, learning_rate=0.1109075,algorithm='SAMME.R'),#适应性提升
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
    'SGDClassifier': SGDClassifier(loss= 'hinge', penalty='l1', alpha=1e-4, 
                                   random_state=42, max_iter=8, tol=0.5),
    
    'bagging1'      : BaggingClassifier(MLPClassifier(hidden_layer_sizes=(90, ),
                                                     activation ='tanh',alpha=0.0005),
                                       max_samples=0.5, max_features=0.5) ,
    'bagging2'      : BaggingClassifier(LogisticRegression(C=1.1,penalty='l1', tol=0.01),
                                       max_samples=0.5, max_features=0.5) ,
    'bagging3'      : BaggingClassifier(SVC(kernel="linear", C=0.02,decision_function_shape ='ovo'),
                                       max_samples=0.5, max_features=0.5) 
    
    }
#
#clf=
sfolder = StratifiedKFold(n_splits=10,random_state=0,shuffle=False)
for train, test in sfolder.split(X_train_fs ,train_label):
    print('Train: %s | test: %s' % (train, test))
    print(" ")

##clf.fit(train_data, train_label)
#for name, clf in classifiers.items():
##    scores = cross_val_score(clf, train_data, train_label,cv=10)
#    scores = cross_val_score(clf, train_data, train_label,cv=10)
##    KFold(n_splits=2, random_state=None, shuffle=False)
#    print(name,'\t--> ',scores.mean())
#    
#qw=fs.get_support()
##步骤5：结果评估，此处直接提供分别用于训练集和测试集上的结果，以及混淆矩阵
#    clf.fit(train_data1, train_label1)
#    train_pred = clf.predict(train_data1)
#    train_acc_sum = (train_pred == train_label1).sum()
#    train_acc = train_acc_sum/len(train_pred)
#    train_cm = confusion_matrix(train_label, train_pred)
#    print("训练集的分类正确率: %f%%" %(train_acc*100))
#    print('训练集的混淆矩阵：')
#    print(train_cm)
#    test_pred = clf.predict(test_data1)
#    test_acc_sum = (test_pred == test_label1).sum()
#    test_acc = test_acc_sum/len(test_pred)
#    test_cm = confusion_matrix(test_label, test_pred)
#    print("测试集的分类正确率: %f%%" %(test_acc*100))
#    print('测试集的混淆矩阵：')
#    print(test_cm)
