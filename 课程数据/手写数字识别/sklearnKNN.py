import numpy as np
from os import listdir
from sklearn import neighbors

def img2vector(fileName):
    retMat = np.zeros([1024],int)
    fr = open(fileName)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32+j] = lines[i][j]
    return retMat

def readDataSet(path):
    fileList = listdir(path)
    numFile = len(fileList)
    dataSet = np.zeros()