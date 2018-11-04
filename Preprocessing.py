# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 22:33:51 2018
预处理的起因：
1，	对很多机器学习算法来说，数据符合正态分布是其进行统计估计的一个前提；
2，	部分特征数值，要么高低差异太大，要么集中度太高。如果做适当的预处理，
   能够扩大各个类别在某个特征维度上的区分度。
推荐阅读：Should I normalize/standardize/rescale the data?  
         http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html

Scikit-Learn提供了专门的Preprocessing模块来预处理数据，我们在这里提供了
这个模块的五种预处理方法

@author: Xie Lingyun
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#读取arff文件的数据
path = 'E:/Data/music_emotion.csv'
data = pd.read_csv(path,dtype = np.float64, converters={'class':int},index_col=0)
train_data = data.values
train_label = np.array(data.index)

#特征预处理（归一化或者变换尺度到某个范围），以下列出5种方法：

#方法1：正态化--减均值，除以标准差，得到均值为0，方差为1的标准正态分布
train_data = preprocessing.scale(train_data)

#方法2：上下限范围变换
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,8))
#train_data = min_max_scaler.fit_transform(train_data)

#方法3：按最大值归一化
#max_abs_scaler = preprocessing.MaxAbsScaler()
#train_data = max_abs_scaler.fit_transform(train_data)

#方法4：分位数变换--一种非线性变换，按照统计中的分位数范围，
#把数据映射为0-1区间的均匀分布‘uniform’或正态分布‘normal’
#quantile_transformer = preprocessing.QuantileTransformer(
#        output_distribution='normal',random_state=0)
#train_data = quantile_transformer.fit_transform(train_data)

#方法5：标准化--L1（除以L1范数）或L2（除以L2范数）
#train_data = preprocessing.normalize(train_data, 'l2', axis=0)


#拆分操作：以stratification的方式将数据拆分为训练集和测试集，测试集的比例可以通过test_size设定
#此处要注意一个参数：random_state，如果不设置它，每次运行split会随机生成不同的train和test
#如果给它指定任意一个整数，那么每次运行split得到的分组是不变的，便于算法变动时，
#在同一个数据集上看效果的变化
train_data, test_data, train_label, test_label = train_test_split(
        train_data, train_label, test_size=0.25, stratify=train_label, random_state=0)

#训练分类器，此处用Stochastic Gradient Descent (SGD) 分类器为例
#选用这个分类器，是因为它对数据的Scaling非常敏感，尤其钟爱正态化的数据，所以特地用在数据预处理这个部分
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000, tol=1e-3)
clf.fit(train_data, train_label)

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
print("\n测试集的分类正确率: %f%%" %(test_acc*100))
print('测试集的混淆矩阵：')
print(test_cm)
