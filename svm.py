# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

def svm():
     print('start')
     dataset = np.loadtxt("/Users/xingtao/PycharmProjects/zhongxing/train1.csv", delimiter=',')
     # df = pd.DataFrame(dataset)
     # df[df.columns[-1]] = df[df.columns[-1]].shift(-1)
     # dataset = df.values

     label = dataset[:, -1]
     print(label.shape)
     feature = dataset[:, :-1]
     print(label.shape)

     x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=0)
     return svm_c(x_train, x_test, y_train, y_test)

def svm_c(x_train, x_test, y_train, y_test):
     # rbf核函数，设置数据权重
     svc = SVC(kernel='rbf', class_weight='balanced',)
     c_range = np.logspace(-5, 15, 11, base=2)
     gamma_range = np.logspace(-9, 3, 13, base=2)
     # 网格搜索交叉验证的参数范围，cv=3,3折交叉
     param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
     grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
     # 训练模型
     clf = grid.fit(x_train, y_train)
     y_svm = clf.predict(x_test)
     # 计算测试集精度
     #score = grid.score(x_test, y_test)
     #print(precision_score(y_test, y_svm, average='macro', zero_division=0))
     #print(recall_score(y_test, y_svm, average='macro'))
     return precision_score(y_test, y_svm, average='macro', zero_division=0),recall_score(y_test, y_svm, average='macro')

if __name__ == "__main__":
     svm()
