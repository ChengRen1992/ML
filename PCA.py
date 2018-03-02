#PCA降维鸢尾花
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
#读取数据，默认return_X_y=False,以字典形式输出data和target;当return_X_y=Ture,分别返回data和target。
data = load_iris()
y = data.target
x = data.data
#加载PCA算法，设置降维后主成分数目为2.
pca = PCA(n_components=2)
#对原始数据进行降维，保存在reduced_X中
reduced_X = pca.fit_transform(x)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
#按照鸢尾花的类别将降维后的数据点保存在不同的列表中。
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
#降维后数据可视化，以散点图表示。
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()