import numpy as np
import pandas as pd
from sklearn.datasets import load_boston,load_iris,load_digits
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score   #导入计算交叉验证的函数
from sklearn.naive_bayes import GaussianNB
# b = np.array([[1,2,3],[4,5,6]])
# boston = load_boston()
# iris = load_iris()
# digits = load_digits()
# plt.matshow(digits.images[1])
# plt.show()
# plt.style.use('mystyle')
# fig = plt.figure(figsize=(4,4))
# ax = fig.add_subplot(111)
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.spines['bottom'].set_position(('data',0))
# ax.yaxis.set_ticks_position('left')
# ax.spines['left'].set_position(('data',0))
# ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi])
# x = np.arange(0,2*np.pi,np.pi/100)
# y = np.sin(x)
# plt.plot(x,y)
#
# plt.show()
# clf = DecisionTreeClassifier()
# iris = load_iris()
#
# cross_val_score(clf,iris.data,iris.target,cv=10)
# clf.fit()
# X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
# Y = np.array([1,1,1,2,2,2])
# clf = GaussianNB(priors=None)
# clf.fit(X,Y)
# print(clf.predict([[-0.8,-1]]))
# a = [[2,3]]
# print(a[0][0])
# feature = np.ndarray(shape=(0,41))
# print(feature.shape)
a = np.array([1,2,3])
b = np.array([4,5,6])
plt.plot(a,b,color='red')
plt.show()

