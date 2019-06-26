import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()
onlinetimes = []
f = open('TestData.txt')
for line in f:
    mac = line.split(',')[2]
    onlinetime = int(line.split(',')[6])
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime,onlinetime))
    else:
        onlinetimes[mac2id[mac]]=[(starttime,onlinetime)]

real_X = np.array(onlinetimes).reshape((-1,2))

X = real_X[:,[0]]
db = skc.DBSCAN(eps = 0.01,min_samples=20).fit(X)
labels = db.labels_
print('Labels:')
print(labels)
raito = len(labels[labels[:]==-1])/len(labels)
print("Noise raito:",format(raito,'.2%'))

n_clisters_ = len(set(labels)) - (1 if  -1 in labels else 0)
print("Estimated number of clusters: %d " % n_clisters_)
print("Silhouette coefficient: %0.3f" % metrics.silhouette_score(X,labels))

for i in range(n_clisters_):
    print("cluster",i,':')
    print(list(X[labels == i].flatten()))

plt.hist(X,24)
plt.show()

Y = np.log(1+real_X[:,[1]])
dbs = skc.DBSCAN(eps=0.14,min_samples=10).fit(Y)
labels = db.labels_

print('labels:')
print(labels)
raito = len(labels[labels[:] == -1])/len(labels)
print("Noise raito:",format(raito,'.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated number of clusters:%d" % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(Y,labels))

for i in range(n_clusters_):
    print('Cluster',i,':')
    count = len(Y[labels == i])
    mean = np.mean(real_X[labels == i][:,1])
    std = np.std(real_X[labels ==i][:,1])
    print('\t number of sample:',count)
    print('\t mean of sample : ',format(mean,'.1f'))
    print('\t std of sample : ',format(std,'.1f'))