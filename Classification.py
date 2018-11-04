# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:13:47 2018

分类中最基础的问题是二元分类，很多分类方法也都是先从二类区分开始的，再发展到多分类的。
这样的方法，在从二类问题过渡到多类问题时，一般有两种不同的策略：
1，有的采取把每一类都和其他类单独一一比较，分别训练出分类器。测试数据要通过所有的分类器，
取得在各个分类器上的表现，综合后决定其归属。这种策略叫做“one vs one”（ovo）。
2，也有的采取把其中一类和其他所有类的集合对比，这样就训练出每一类和其他类集合区分开来的分类器。
测试数据通过这些分类器后，综合表现决定归属。这种策略叫做“one vs all”，在scikit-learn里面叫做“one vs rest”（ovr）。

当然，还有的算法，是本来就适用于多类问题的，例如Tree、最近邻、朴素贝叶斯和高斯过程这样的概率模型、
神经网络、判别分析等等。

除此以外，还有集成分类法（sklearn.ensemble），这是一种将多个分类器打包集成以提升分类效果的方法，
像随机森林。

在scikit-learn里面，用于多分类的模型较多。按照以上介绍的内容，我们给出如下代表性的示例：
1，SVM中的SVC：基于libsvm实现的C-支持向量机（采取ovo策略）
2，SVM中的LinearSVC：线性SVM（采取ovr策略）
3，LogisticRegression：逻辑回归分类（采取ovr策略）
4，DecisionTreeClassifier：决策树（经典分类模型）
5，KNeighborsClassifier：最近邻分类（经典分类模型）
6，GaussianNB：高斯朴素贝叶斯（概率分类模型）
7，MLPClassifier：多层感知器（神经网络），这是学习如何调参的好对象
8，RandomForestClassifier：随机森林（集成分类器）

有关scikit-learn的多类分类模型的列表，可参考：http://scikit-learn.org/stable/modules/multiclass.html#multiclass
@author: Xie Lingyun
"""

import numpy as np
import pandas as pd

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

path = 'D:/Data/Arff/music_emotion.csv'
#此处是调用pandas的读取csv格式的函数，中间设定了数据格式，同时把
#标签那一列读为int类型，设定标签列为最开始那一列
data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=0)
mydata = data.values
mylabel = np.array(data.index)
#随机将数据分为训练集和测试集，测试集的比例可以通过test_size设定
train_data, test_data, train_label, test_label = train_test_split(
        mydata, mylabel, test_size=0.2, random_state=0)


#选择不同的分类器，此处均采用默认参数，实际上每个分类器都有调参的余地，最好能深入
#了解一下分类算法的原理和对应的参数，自己手动选择更合适的。MLPClassifier的结果和参数
#关系很大，默认值一般不会有很好的结果
clf = SVC()
#clf = LinearSVC()
#clf = LogisticRegression()
#clf = DecisionTreeClassifier()
#clf = KNeighborsClassifier()
#clf = GaussianNB()
#clf = MLPClassifier()
#clf = RandomForestClassifier()

clf.fit(train_data,train_label)

#结果评估，此处直接提供分别用于训练集和测试集上的结果，以及混淆矩阵
train_pred = clf.predict(train_data)
train_acc_sum = (train_pred == train_label).sum()
train_acc = train_acc_sum/len(train_pred)
train_cm = confusion_matrix(train_label, train_pred)
print("训练集的分类正确率: %f%%" %(train_acc*100))
print('训练集的混淆矩阵：')
print(train_cm)
test_pred = clf.predict(test_data)
test_acc_sum = (test_pred == test_label).sum()
test_acc = test_acc_sum/len(test_pred)
test_cm = confusion_matrix(test_label, test_pred)
print("测试集的分类正确率: %f%%" %(test_acc*100))
print('测试集的混淆矩阵：')
print(test_cm)
