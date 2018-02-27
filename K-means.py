import numpy as np
from sklearn.cluster import KMeans

# 读取文件
def loadData(filePath):
    fr = open(filePath, 'r+', encoding='gbk')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName


if __name__ == '__main__':
    data, cityName = loadData('city.txt')
    km = KMeans(n_clusters=6)                             #n_clusters:用于指定聚类中心的个数
    label = km.fit_predict(data)                          #fit_predict():计算簇中心以及为簇分配序号
    expenses = np.sum(km.cluster_centers_, axis=1)        #聚类中心点的数值相加(即城市的平均花费)
    CityCluster = [[], [], [], [],[],[]]
    for i in range(len(cityName)):                       #将城市按label分成设定的簇，将每个簇中的城市输出
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])