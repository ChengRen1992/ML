#上证指数预测涨跌
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# read_csv:参数一:数据源.encoding:编码格式.parse_dates:第n列解析为日期.index_col:用作索引的列编号
# sort_index:参数一:按0列排,ascending(true)升序,inplace:排序后是否覆盖原数据(data按时间升序排列)
data = pd.read_csv('data\stock.csv',encoding='gbk',parse_dates=[0],index_col=0)
data.sort_index(0,ascending=True,inplace=True)
#dayfeature:选取150天的数据
#featurenum:特征数量
dayfeature = 150
featurenum = 5*dayfeature
x = np.zeros((data.shape[0]-dayfeature,featurenum+1))
y = np.zeros((data.shape[0]-dayfeature))
# print(y.shape)

for i in range(0,data.shape[0]-dayfeature):
    for i in range(0, data.shape[0] - dayfeature):
        x[i, 0:featurenum] = np.array(data[i:i + dayfeature][[u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量']]).reshape((1, featurenum))
        x[i, featurenum] = data.ix[i + dayfeature][u'开盘价']
for i in range(0,data.shape[0]-dayfeature):
    if data.ix[i+dayfeature][u'收盘价']>=data.ix[i+dayfeature][u'开盘价']:
        y[i]=1
    else:
        y[i]=0
clf = svm.SVC(kernel='rbf')
result1 = []
for i in range(5):
    #train_test_split随机划分训练集和测试集，是交叉验证中常用的函数
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
    clf.fit(x_train,y_train)
    result1.append(np.mean(y_test == clf.predict(x_test)))

result2 = cross_val_score(clf,x_train,y_train,cv=10)
print("svm classifier accuacy1:")
print(result1)
print(np.mean(result1))
print("svm classifier accuacy2:")
print(result2)