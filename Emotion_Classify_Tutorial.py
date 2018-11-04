# -*- coding: utf-8 -*-
"""
这是机器学习的入门实例。以音乐情感的带分类特征数据文件为对象，涉及到（1）数据的读取，
（2）数据预处理，(3)特征选择，（4）训练分类器，（5）结果评估 等五个步骤。
需预先安装scikit-learn

@author: Xie Lingyun
"""
import operator
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#步骤1：读取arff文件的数据，该民乐情感数据集共有207个样本，192个特征
#       有四类情感标签：1-anger(55个),2-depress(68个),3-happy(41个),4-easy(43个)
f = open('D:/Data/Arff/music_emotion.arff')
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

#步骤2：特征预处理，此处将特征数据范围变换为[0,8]区间
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,8))
train_data = min_max_scaler.fit_transform(train_data)

#步骤3：特征选择，做ANOVA分析，以F值高低排序，选择前K个特征
train_data = SelectKBest(k=60).fit_transform(train_data, train_label)

#拆分操作：随机将数据拆分为训练集和测试集，测试集的比例可以通过test_size设定
train_data, test_data, train_label, test_label = train_test_split(
        train_data, train_label, test_size=0.25, random_state=0)

#步骤4：训练分类器，此处用svm分类器为例
clf = svm.SVC(kernel='linear')
clf.fit(train_data, train_label)

#步骤5：结果评估，此处直接提供分别用于训练集和测试集上的结果，以及混淆矩阵
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
