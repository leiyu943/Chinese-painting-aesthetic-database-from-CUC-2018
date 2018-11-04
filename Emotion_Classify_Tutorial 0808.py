# -*- coding: utf-8 -*-
"""
这是机器学习的入门实例。以音乐情感的带分类特征数据文件为对象，涉及到（1）数据的读取，
（2）数据预处理，(3)特征选择，（4）训练分类器，（5）结果评估 等五个步骤。
需预先安装scikit-learn

@author: Xie Lingyun
"""
import operator
from sklearn import svm
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# meta-estimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression


#步骤1：读取arff文件的数据，该民乐情感数据集共有207个样本，192个特征
#       有四类情感标签：1-anger(55个),2-depress(68个),3-happy(41个),4-easy(43个)
f = open('D:/feature/feature0811 - 158 features.arff')
#
lines = f.readlines()
count = 0
train_data = []
train_label = []
#
#
#with open('D:/feature/feature0720-5lei.csv', 'rb') as csvfile:
#    train_data = csv.reader(csvfile, delimiter=',', quotechar='|')

for l in lines:
    if count == 1:
        content = l.split(',')#获得该行数据
        tmp = len(content)-1#获得数据长度
        tmplabel = content[tmp].strip('\n') #获取最末尾的标签 并移除数据末尾的'\n'
        tmplabel = int(tmplabel) #将标签转化为
        if tmplabel == 6:
            tmplabel = 0
        train_label.append(tmplabel)
        content.remove(content[tmp])
        content = list(map(lambda x: float(x), content))
        train_data.append(content)
    if operator.eq(l,'@data\n') == True:
        count = 1  

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

#data=csv.reader(open('D:/feature/feature0720-5lei.csv','rb'))
#data.readlines
#步骤2：特征预处理，此处将特征数据范围变换为[0,8]区间
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,10))
train_data = min_max_scaler.fit_transform(train_data)

#步骤3：特征选择，做ANOVA分析，以F值高低排序，选择前K个特征
X_train_fs = SelectKBest(k=50).fit_transform(train_data, train_label)
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=80)
X_train_fs = fs.fit_transform(train_data, train_label)

#拆分操作：随机将数据拆分为训练集和测试集，测试集的比例可以通过test_size设定
train_data1, test_data, train_label1, test_label = train_test_split(
        train_data, train_label, test_size=0.25, random_state=0)

#步骤4：训练分类器，此处用svm分类器为例
#clf=AdaBoostClassifier(n_estimators=100)#35.7,25
#clf = DecisionTreeClassifier()#max_depth=5,99.73,30or26or25
#clf = RandomForestClassifier(n_estimators=10)#95,30;92;32;94,33
#clf = svm.SVC(kernel='linear')#51,30;51,30
#clf=KNeighborsClassifier(11)#35,8,38;11,40,37
classifiers = {
#    'SVM':
    'KNN         ': KNeighborsClassifier(20),
    'SVC1        ': SVC(kernel="linear", C=0.025),
    'SVC2        ': SVC(gamma=2, C=1),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=10, max_depth=5, 
                                           max_features=1),  # clf.feature_importances_
    'ExtraTrees  ': ExtraTreesClassifier(n_estimators=10, max_depth=None),  #
    'AdaBoost    ': AdaBoostClassifier(n_estimators=5),#适应性提升
    'Gradient Boo': GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                               max_depth=1, random_state=0), #梯度提升数
    'GaussianNB  ': GaussianNB(),#贝叶斯
    'LinearDisc  ': LinearDiscriminantAnalysis(),#线性判别分析          
    'MultinomialNB': MultinomialNB(),
    'SGDClassifier': SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                   random_state=42, max_iter=5, tol=None),
    'logistic re ': LogisticRegression(C=1.0, penalty='l1', tol=0.01),#
    'QuadraticDisc': QuadraticDiscriminantAnalysis(),#二次判别分析  
    'MLP         ' : MLPClassifier(hidden_layer_sizes=(100, ))
    
    }
#
#clf=
#
#clf.fit(train_data, train_label)
for name, clf in classifiers.items():
#    scores = cross_val_score(clf, train_data, train_label,cv=10)
    scores = cross_val_score(clf, X_train_fs, train_label,cv=10)
    print(name,'\t--> ',scores.mean())
    
#
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
