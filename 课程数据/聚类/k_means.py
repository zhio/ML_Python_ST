import numpy as np
from sklearn.cluster import KMeans

def loadData(filePath):
    fr = open(filePath,'r+',encoding='gbk')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(',')
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1,len(items))])
    return retData,retCityName

if __name__ == '__main__':
    data,cityName = loadData('city.txt') #读取数据
    km = KMeans(n_clusters=4)  #选择四个聚类点
    label = km.fit_predict(data) #已经聚类完成，划分成四类
    expenses = np.sum(km.cluster_centers_,axis=1)
    print(expenses)
    CityClister = [[],[],[],[]]
    for i in range(len(cityName)):
        CityClister[label[i]].append(cityName[i])
    for i in range(len(CityClister)):
        print("Expenses:%.2f" % expenses[i])
        print(CityClister[i])