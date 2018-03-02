#K-means进行图像分割
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans

def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    m, n = img.size
    for i in range(m):
        for j in range(n):
            #输出图像的R,G,B，范围在0-1
            x, y, z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    #以矩阵形式返回data，以及图片大小
    return np.mat(data),m,n

imgData,row,col = loadData('data\starbucks.jpg')
print(imgData,row,col)
#聚类获得每个像素所属的类别
label = KMeans(n_clusters=4).fit_predict(imgData)
label = label.reshape([row,col])
print(label.shape)
#创建一张新的灰度图保存聚类后的结果
pic_new = image.new('L',(row,col))
#根据所属类别向图片中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
pic_new.save(r'data\result-bull-4.jpg','JPEG')                #r表示不转义

