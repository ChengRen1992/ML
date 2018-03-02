#DBSCAN学生上网时长分析
from numpy.random import RandomState                  #用于创建随机种子
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn import decomposition                     #加载PCA算法

n_row, n_col = 2, 3
n_components = n_row * n_col                          #设置提取的特征数目
image_shape = (64, 64)                                #设置图片大小为64x64
dataset = fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))
faces = dataset.data                                  #加载数据，并打乱数据
# print(faces)
# print(faces[:n_components].shape)
#设置图像的展示方式
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))    #创建图片，并指定图片大小（英寸）
    plt.suptitle(title, size=16)                      #设置标题及字号大小
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i+1)                #选择画制的子图
        vmax = max(comp.max(), -comp.min())
        #对数值归一化，并以灰度图形式显示
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest', vmin=-vmax, vmax=vmax)
        #去除子图的坐标轴标签
        plt.xticks(())
        plt.yticks(())
    #对子图位置及间隔调整
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)
    # plt.show()
plot_gallery("First centered 0livetti faces", faces[:n_components])
#创建特征提取的对象，使用PCA作为对比
estimators = [('Eigenfaces - PCA using randomized SVD',decomposition.PCA(n_components=n_components, whiten=True)),('Non-negative compinents - NMF',decomposition.NMF(n_components=n_components, init='nndsvda',tol=5e-3))]
#分别调用PCA和NMF，estimator表示提取对象
for name, estimator in estimators:
    print('Extracting the top %d %s...' %(n_components, name))
    print(faces.shape)
    estimator.fit(faces)                                #调用PCA或NMF提取特征
    components_ = estimator.components_                 #获取提取的特征
    plot_gallery(name, components_[:n_components])
plt.show()