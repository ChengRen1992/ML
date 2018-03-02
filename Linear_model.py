#对房屋成交信息建立简单线性回归方程，并依据回归方程对房屋价格进行预测(线性关系)
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

datasets_X = []                    #房屋面积
datasets_Y = []                    #房屋价格
fr = open('data\prices.txt','r')
next(fr)                           #忽略prices.txt文件的第一行
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))

length = len(datasets_X)
#将datasets_X转化为数组，并变为二维，以符合线性回归拟合函数输入参数要求
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
minX = min(datasets_X)
maxX = max(datasets_X)
#以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图。
X = np.arange(minX,maxX).reshape([-1,1])
linear = linear_model.LinearRegression()
#fit函数中，dataset_X需要是二维，datasets_Y是一维
linear.fit(datasets_X,datasets_Y)


plt.scatter(datasets_X,datasets_Y, color='red')      #画散点图
plt.plot(X,linear.predict(X), color='blue')          #画出回归直线
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()