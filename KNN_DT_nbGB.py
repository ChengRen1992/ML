#KNN，DT，GB模型评价指标
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer                             #预处理模块Imputer
from sklearn.model_selection import train_test_split                 # 导入自动生成训练集和测试集的模块 train_test_split
from sklearn.metrics import classification_report                     # 导入预测结果评估模块 classification_report
import numpy as np
import pandas as pd
#加载数据
def load_dataset(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))
    for file in feature_paths:
        #使用逗号分隔符读取特征数据，用问号替换缺失值，文件中不包含表头
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        #使用平均值补全缺失值，axis=0表示列
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(df)                                       #fit()函数用于训练预处理器
        df = imp.transform(df)                            #transform()函数用于生成预处理结果
        feature = np.concatenate((feature,df))             #将数组feature与数组df进行拼接。
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label,df))
    label = np.ravel(label)                               #将多维数组降为一维
    return feature, label                                #将特征集合feature和标签集合label返回

if __name__ == '__main__':
    featurePaths = ['data\A.feature', 'data\B.feature', 'data\C.feature', 'data\D.feature', 'data\E.feature']
    labelPaths = ['data\A.label', 'data\B.label', 'data\C.label', 'data\D.label', 'data\E.label']
    x_train,y_train = load_dataset(featurePaths[:4],labelPaths[:4])
    x_test,y_test = load_dataset(featurePaths[4:],labelPaths[4:])
    #使用全量数据作为训练集，train_test_split中设置test_size=0将数据随机打乱。
    x_train,x_,y_train,y_ = train_test_split(x_train,y_train,test_size=0.0)

    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train,y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print(answer_knn)
    print('Prediction done')

    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train,y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print(answer_dt)
    print('Prediction done')

    print('Start train Bayes')
    gnb = GaussianNB().fit(x_train,y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print(answer_gnb)
    print('Prediction done')
    #评估：主要指标有:精确度，召回率和F1值
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))
