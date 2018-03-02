#对房屋成交信息建立多项式回归方程，并依据回归方程对房屋价格进行预测
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

datasets_X = []
datasets_Y = []
fr = open('data\prices.txt','r')
next(fr)
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))
length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length,1])
datasets_Y = np.array(datasets_Y)
minX = min(datasets_X)
maxX  =max(datasets_X)
X = np.arange(minX,maxX).reshape([-1,1])
#degree=2表示多项式的阶数
poly_reg = PolynomialFeatures(degree=2)
#将datasets_X进行多项式特征转化
X_poly = poly_reg.fit_transform(datasets_X)
print(X_poly,type(X_poly),X_poly.shape)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly,datasets_Y)

plt.scatter(datasets_X, datasets_Y, color='red')
#画线
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()