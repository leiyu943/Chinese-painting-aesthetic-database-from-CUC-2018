"""
Created on Mon Jul 16 2018
调用scikit-learn的分类模型来对数据进行分类
@author: Xie Lingyun
"""
import numpy as np
from scipy.io import arff
from sklearn import tree
#from sklearn.feature_selection import SelectKBest, chi2

#读取训练集的arff文件
data, meta = arff.loadarff('E:/PythonCode/mxy/train.arff')
sample_num = len(data)
feature_num = len(data[0])-1
tmpdata = []
for n in range(len(data)):
    tmpdata.append(list(data[n]))
train_data = []
train_label = np.ones(sample_num, dtype=np.int16)
for i in range(sample_num):
    train_data.append(tmpdata[i][0:feature_num])
    train_label[i] = tmpdata[i][feature_num]

#读取测试集的arff文件
data, meta = arff.loadarff('E:/PythonCode/mxy/test.arff')
sample_num = len(data)
feature_num = len(data[0])-1
tmpdata = []
for n in range(len(data)):
    tmpdata.append(list(data[n]))
test_data = []
test_label = np.ones(sample_num, dtype=np.int16)
for i in range(sample_num):
    test_data.append(tmpdata[i][0:feature_num])
    test_label[i] = tmpdata[i][feature_num]

#引入决策树分类模型进行分类器训练,并用测试集测试效果
clf = tree.DecisionTreeClassifier()
result = clf.fit(train_data, train_label)

#分类结果评估
train_pred = clf.predict(train_data)
train_acc_sum = (train_pred == train_label).sum()
train_acc = train_acc_sum/len(train_pred)
test_pred = clf.predict(test_data)
test_acc_sum = (test_pred == test_label).sum()
test_acc = test_acc_sum/len(test_pred)
print("Train Set Result: %f%%" %(train_acc*100))
print("Test Set Result: %f%%" %(test_acc*100))

#X_new = SelectKBest().fit_transform(feature_data, label)



