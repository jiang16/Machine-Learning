#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *


#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


#批量梯度算法
def gradAscent(dataMatIn, classLabels):
    #转换为numpy的mat矩阵
    dataMatrix = mat(dataMatIn)
    #转换为numpy的mat矩阵，并进行转置
    labelMat = mat(classLabels).transpose()
    #返回dataMatrix的大小，m为行数，n为列数
    m, n = shape(dataMatrix)
    #移动步长，即学习速率，控制更新的幅度
    alpha = 0.001
    #最大迭代次数
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        #这里h为列向量，元素个数（维数）为样本个数
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        #梯度上升矢量化公式
        weights = weights + alpha * dataMatrix.transpose() * error
    #返回回归系数数组
    return weights


#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    #返回dataMatrix的大小，m为行数，n为列数
    m, n = shape(dataMatrix)
    #参数初始化，这里系数为一维数组
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #降低alpha的大小
            alpha = 4 / (1.0 + i + j) + 0.01
            #随机选取样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            #选择随机选取的一个样本，计算h
            h = sigmoid(sum(dataMatrix[i] * weights))
            #计算误差
            error = classLabels[i] - h
            #更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            #删除已经使用的样本
            del(dataIndex[randIndex])
    #返回回归系数数组
    return weights


#分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0


#使用Logistic回归分类器做预测
def colicTest():
    #打开训练集文件
    frTrain = open('horseColicTraining.txt')
    #打开测试集文件
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    #使用批量梯度上升算法训练
    trainWeights = gradAscent(array(trainingSet), array(trainingLabels))
    #使用改进的随机梯度上升算法训练
    #trainWeights = stocGradAscent1(array(trainingSet), array(trainingLabels), 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #计算错误次数
        if int(classifyVector(lineArr, trainWeights)) != int(currLine[-1]):
            errorCount += 1
    #计算错误率
    errorRate = float(errorCount) / numTestVec
    print 'the error rate of this test is: %f' % errorRate
    return errorRate


if __name__ == '__main__':
    colicTest()