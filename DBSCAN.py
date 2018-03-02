import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()
onlinetimes = []
f = open('data\TestData.txt',encoding='utf-8')
for line in f:
    mac = line.split(',')[2]
    onlinetime = int(line.split(',')[6])
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac]=len(onlinetimes)
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime,onlinetime)]
#289x2
real_X = np.array(onlinetimes).reshape((-1,2))
# print(real_X.shape)
# starttime
X = real_X[:,0:1]
# print('X')
# print(X)
#调用DBSCAN方法进行训练
#eps:两个样本被看作邻居节点的最大距离。min_samples:簇的样本数。metric:距离计算方式。labels:每个数据的簇标签
db = skc.DBSCAN(eps=0.01,min_samples=20).fit(X)
labels = db.labels_
print('Labels:')
print(labels)
#噪声数据的比例
raito = len(labels[labels[:] == -1])/len(labels)
print('Noise raito:',format(raito,'.2%'))
#簇的个数
n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)
print('Estimated number of clusters: %d' %n_clusters_)
#打印聚类效果评价指标
print('Silhouette coefficient: %0.3f' %metrics.silhouette_score(X,labels))
#打印各簇标号及簇内数据
for i in range(n_clusters_):
    print('Cluster ',i,':')
    print(list(X[labels==i].flatten()))
'''
首先要理清楚一个概念，直方图与条形图。
直方图与条形图的区别：
条形图是用条形的长度表示各类别频数的多少，其宽度（表示类别）则是固定的；
直方图是用面积表示各组频数的多少，矩形的高度表示每一组的频数或频率，宽度则表示各组的组距，因此其高度与宽度均有意义。
由于分组数据具有连续性，直方图的各矩形通常是连续排列，而条形图则是分开排列。
条形图主要用于展示分类数据，而直方图则主要用于展示数据型数据。
'''
#直方图展示，第二个参数是柱子宽一些还是窄一些，越大越窄越密
plt.hist(X,30)
#设置x的范围，（0，24）
plt.xlim([0,24])
#设置x轴刻度
plt.xticks([0,5,10,15,20,25])
plt.show()