#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *


#加载数据
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


#使用局部加权线性回归计算回归系数w
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    #创建权重对角矩阵
    weights = mat(eye((m)))
    #遍历数据集计算每个样本的权重
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print '矩阵为奇异矩阵，不能求逆'
        return
    #计算回归系数
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


#局部加权线性回归测试
def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


#简单线性回归计算回归系数w
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print '矩阵为奇异矩阵，不能求逆'
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


#误差大小评估函数
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()


if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    print '训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:'
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print 'k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T)
    print 'k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T)
    print 'k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T)
    print
    print '训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:'
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print 'k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T)
    print 'k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T)
    print 'k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T)
    print
    print '训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:'
    print 'k=1时,误差大小为:', rssError(abY[100:199], yHat1.T)
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = mat(abX[100:199]) * ws
    print '简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A)