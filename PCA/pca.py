#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(filename):
    fr = open(filename)
    datArr = [map(float, line.strip().split('\t')) for line in fr.readlines()]
    return datArr


def pca(dataMat, N = 100):
    meanVals = mean(dataMat, axis = 0)
    #去掉平均值
    meanRemoved = dataMat - meanVals
    #计算协方差矩阵
    covMat = cov(meanRemoved, rowvar = 0)
    #计算协方差矩阵的特征值和特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    #将特征值从小到大排序
    eigValInd = argsort(eigVals)
    #保留最大的前N个特征向量
    eigValInd = eigValInd[: -(N + 1): -1]
    redEigVects = eigVects[:, eigValInd]
    #将数据转换到上述N个特征向量构建的新空间中
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


if __name__ == '__main__':
    dataMat = mat(loadDataSet('testSet.txt'))
    lowDMat, reconMat = pca(dataMat, 1)
    
    #将降维后的数据和原始数据一起绘制出来
    plt.scatter(dataMat[:, 0].tolist(), dataMat[:, 1].tolist(), color = 'b', s = 90, marker = '^')
    plt.scatter(reconMat[:, 0].tolist(), reconMat[:, 1].tolist(), color = 'r', s = 50, marker = 'o')
    plt.show()