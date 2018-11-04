# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:34:48 2018

这部分代码是利用scikit-learn的feature selection模块，介绍机器学习的特征选择
周志华在《机器学习》中介绍过，特征选择有三类方法：过滤式；包裹式；嵌入式。在scikit-learn里面都有涉及。
过滤式是与分类器无关的特征选择，在scikit-learn里面主要提供了单变量因素统计测试方法来进行这一类特征选择；
包裹式是选择不同的特征子集，用分类器来评估效果，再选择效果最好的特征集合。scikit-learn有RFE算法等；
嵌入式是指特征选择与分类模型训练合为一体，在训练优化的同时也选择了更合适的特征，常见于各种正则化算法，
scikit-learn里面有lasso，elastic net和ridge回归。这类方法普遍是用于回归而不是分类。分类方法里面，
在训练分类器的同时就对特征进行评估的有LinearSVM，Tree等等。
我们在这三类方法里面各选一个或几个来实现。

推荐阅读：An Introduction to Feature Selection
    链接：https://machinelearningmastery.com/an-introduction-to-feature-selection/
@author: Xie Lingyun
"""
import numpy as np
import pandas as pd
from sklearn import svm, linear_model
from sklearn.feature_selection import SelectKBest, SelectFpr, f_classif, RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

#读取arff文件的数据
path = 'E:/Data/music_emotion.csv'
data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=0)
train_data = data.values
train_label = np.array(data.index)


#1，过滤式特征选择方法：SelectFpr--基于False Positive Rate，FPR（误报率）测试的特征选择
#                  SelectKBest--基于假设检验的P值排名取前K个特征
#new_data = SelectFpr(f_classif).fit_transform(train_data, train_label)
#new_data = SelectKBest(f_classif, k=60).fit_transform(train_data, train_label)


#2，包裹式特征选择方法：recursive feature elimination（rfe）--利用有特征重要度评估功能
#的分类模型（这类模型的特点就是会有coef_或者feature_importances_属性），从特征完整集合
#开始，训练并评估，把排名靠后的n个（由step参数指定）特征去掉，然后再训练评估...反复进行，
#直到达到指定的特征个数（由n_features_to_select参数指定）为止。
#我在这里选择了svm中的linearsvc模型，还能承担类似任务的有SGDClassifier，DecisionTreeClassifier,
#以及MultinomialNB,LinearDiscriminantAnalysis等
#svc = svm.LinearSVC()
#rfe = RFE(estimator=svc, n_features_to_select=60, step=1)
#new_data = rfe.fit_transform(train_data, train_label)
#ranking = rfe.ranking_ #特征排名，被选中的排名都是1


#3，嵌入式特征选择方法：这类正则化方法一般用于回归。但是它有稀疏化特征的能力，所以其实也可以用于
# 分类模型的特征选择，只要配上一个合适的线性分类器，比如LinearSVC，或者LogisticRegression。当然，
# 这样的做法和所谓的嵌入式就不匹配了，只能归类到包裹式。我们在这里用Tree分类器来实现特征选择，
# 因为Tree分类方法本身就是符合嵌入式特征选择要求的
clf = ExtraTreesClassifier().fit(train_data, train_label)
#可以把上一行用这个代替：clf = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(train_data, train_label) 
#这是用LinearSVC加上L1正则化方法实现的嵌入式特征选择

model = SelectFromModel(clf,threshold='median',prefit=True) #threshold参数决定了特征选择的阈值
new_data = model.transform(train_data)


#分类以及评估
clf1 = svm.LinearSVC()
clf1.fit(train_data, train_label)
BeforeFeatureNum = len(train_data[1])
pred = clf1.predict(train_data)
acc_sum = (pred == train_label).sum()
acc = acc_sum/len(pred)
print("特征选择前 特征个数：%d, 分类正确率: %f%%" %(BeforeFeatureNum,acc*100))
clf2 = svm.LinearSVC()
clf2.fit(new_data, train_label)
AfterFeatureNum = len(new_data[1])
pred = clf2.predict(new_data)
acc_sum = (pred == train_label).sum()
acc = acc_sum/len(pred)
print("特征选择后 特征个数：%d, 分类正确率: %f%%" %(AfterFeatureNum,acc*100))
