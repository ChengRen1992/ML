#根据已有的数据创建多项式特征，使用岭回归模型代替一般线性模型，对车流量的信息进行多项式回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

data = pd.read_csv('data\Ridge.csv',index_col=0,header=0)
data = np.array(data)
# print(np.where(data[:,4]==20))
# plt.plot(data[:,4])
# plt.show()
#x表示属性，即HR,WEEK_DAY等，y表示TRAFFIC_COUNT
x = data[:,:4]
y = data[:,4]
#用于创建最高次数6次的多项式特征
poly = PolynomialFeatures(6)
#将x进行多项式特征转化
X = poly.fit_transform(x)
#将所有数据划分维训练集和测试集，test_size表示测试集的比例，random_state：是随机数的种子。
'''
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
'''
train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X,y,test_size=0.3,random_state=0)
#alpha:正则化因子，对应于损失函数中的𝜶；fit_intercept:表示是否计算截距;slove:设置计算参数的方法
clf = Ridge(alpha=1.0,fit_intercept=True)
#调用fit函数使用训练集训练回归器
clf.fit(train_set_x,train_set_y)
#利用测试集计算回归曲线的拟合优度,拟合优度，用于评价拟合好坏，最大为1，无最小值，当对所有输入都输出同一个值时，拟合优度为0
clf.score(test_set_x,test_set_y)

start = 200
end = 300
# print(y[200:300])
y_pre = clf.predict(X)
time = np.arange(start,end)
plt.plot(time,y[start:end],'b',label='real')
plt.plot(time,y_pre[start:end],'r',label='predict')
plt.legend(loc='upper left')
plt.show()