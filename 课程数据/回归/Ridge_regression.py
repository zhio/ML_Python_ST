import numpy as np
from sklearn.linear_model import Ridge
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = np.genfromtxt('岭回归.csv',delimiter=',')
data = np.delete(data,0,axis=0)
data = np.delete(data,0,axis=1)

X = data[:,:4]
y = data[:,4]
poly = PolynomialFeatures(6)
X = poly.fit_transform(X)

train_set_X,test_set_X,train_set_y,test_set_y = model_selection.train_test_split(X,y,test_size=0.3,random_state=0)
clf = Ridge(alpha=1.0,fit_intercept=True)
clf.fit(train_set_X,train_set_y)
clf.score(test_set_X,test_set_y)

start = 100
end = 200
y_pre = clf.predict(X)
time = np.arange(start,end)
plt.plot(time,y[start:end],'b',label = 'real')
plt.plot(time,y_pre[start:end],'r',label = 'predict')

plt.legend(loc='upper left')
plt.show()