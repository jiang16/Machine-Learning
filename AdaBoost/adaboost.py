#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *


#自适应数据加载函数
def loadDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


#单层决策树分类函数
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    #初始化retArray为1
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        #如果小于等于阈值，则赋值为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        #如果大于阈值，则赋值为-1
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


#找到数据集上最佳的单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m, 1)))
    #最小误差初始化为无穷大
    minError = inf
    #遍历所有特征
    for i in range(n):
        #找到特征中最小的值和最大值
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        #计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            #less than和greater than
            for inequal in ['lt', 'gt']:
                #计算阈值
                threshVal = rangeMin + j * stepSize
                #计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                #初始化误差矩阵
                errArr = mat(ones((m, 1)))
                #分类正确的赋值为0
                errArr[predictedVals == labelMat] = 0
                #计算误差
                weightedError = D.T * errArr
                #找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


#基于单层决策树的Adaboost训练过程
def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    #初始化权重向量
    D = mat(ones((m, 1)) / m)
    #记录每个数据点的类别估计累计值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        #构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print 'D:', D.T
        #计算弱学习算法权重alpha，使error不等于0,因为分母不能为0
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        #存储弱学习算法权重
        bestStump['alpha'] = alpha
        #存储单层决策树
        weakClassArr.append(bestStump)
        print 'classEst:', classEst.T
        #计算e的指数项
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        #根据样本权重公式，更新样本权重
        D = D / D.sum()
        #计算AdaBoost误差
        aggClassEst += alpha * classEst
        print 'aggClassEst:', aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print 'total error:', errorRate
        #误差为0,退出循环
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst


#AdaBoost分类函数
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    #遍历所有分类器，进行分类
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        #print aggClassEst
    return sign(aggClassEst)


if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr)
    testArr, testLables = loadDataSet('horseColicTest2.txt')
    print weakClassArr
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = mat(ones((len(dataArr), 1)))
    print '训练集的错误率: %.3f%%' % float(errArr[predictions != mat(labelArr).T].sum() / len(dataArr) * 100)
    predictions = adaClassify(testArr, weakClassArr)
    errArr = mat(ones((len(testArr), 1)))
    print '测试集错误率: %.3f%%' % float(errArr[predictions != mat(testLables).T].sum() / len(testArr) * 100)