#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import random


#读取数据
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


#数据结构，维护所有需要操作的值
class optStruct:
    #初始化，参数：数据矩阵，数据标签，松弛变量，容错率，包含核函数信息的元组(第一个参数存放核函数类别，第二个参数存放核函数的参数)
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn                     #数据矩阵
        self.labelMat = classLabels            #类别标签
        self.C = C                             #松弛变量
        self.tol = toler                       #容错率
        self.m = shape(dataMatIn)[0]           #数据矩阵行数
        self.alphas = mat(zeros((self.m, 1)))  #根据数据矩阵行数初始化alpha参数为0
        self.b = 0                             #初始化参数b为0
        self.eCache = mat(zeros((self.m, 2)))  #根据矩阵行数初始化误差缓存，第一列为是否有效的标志位，第二列为实际的误差E的值
        self.K = mat(zeros((self.m, self.m)))  #初始化核K
        for i in range(self.m):                #计算所有数据的核K
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)


#核转换函数
def kernelTrans(X, A, kTup):
    m = shape(X)[0]
    K = mat(zeros((m, 1)))
    #线性核，只进行内积
    if kTup[0] == 'lin': K = X * A.T
    #高斯核函数，根据高斯核函数公式进行计算
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        #计算高斯核K
        K = exp(K / (-1 * kTup[1] ** 2))
    else: raise NameError('核函数无法识别')
    return K


#计算误差
def calcEK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    EK = fXk - float(oS.labelMat[k])
    return EK


#随机选择alpha_j的索引值
def selectJrand(i, m):
    j = i
    #选择一个不等于i的j
    while(j == i):
        j = int(random.uniform(0, m))
    return j


#内循环启发方式
def selectJ(i, oS, Ei):
    #初始化
    maxK = -1; maxDeltaE = 0; Ej = 0
    #根据Ei更新误差缓存
    oS.eCache[i] = [1, Ei]
    #返回误差不为0的数据索引值
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    #有不为0的误差
    if len(validEcacheList) > 1:
        #遍历，找到最大的误差Ek
        for k in validEcacheList:
            #不计算i
            if k == i: continue
            #计算Ek
            Ek = calcEK(oS, k)
            #计算|Ei-Ek|
            deltaE = abs(Ei - Ek)
            #找到maxDeltaE
            if deltaE > maxDeltaE:
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    #没有不为0的误差
    else:
        #随机选择alpha_j的索引值
        j = selectJrand(i, oS.m)
        #计算Ej
        Ej = calcEK(oS, j)
        return j, Ej


#计算Ek，并更新误差缓存
def updateEk(oS, k):
    Ek = calcEK(oS, k)
    oS.eCache[k] = [1, Ek]


#修剪alpha_j
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj


#完整优化的SMO算法
def innerL(i, oS):
    #步骤1：计算误差Ei
    Ei = calcEK(oS, i)
    #优化alpha，设定一定的容错率
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        #使用内循环启发方式选择alpha_j，并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        #保存更新前的alpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        #步骤2：计算上下界L和H
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print 'L == H'
            return 0
        #步骤3：计算eta
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print 'eta >= 0'
            return 0
        #步骤4：更新alpha_j
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        #步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        #更新Ej至误差缓存
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print 'alpha_j变化太小'
            return 0
        #步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        #更新Ei至误差缓存
        updateEk(oS, i)
        #步骤7：更新b1和b2
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        #步骤8：根据b1和b2更新b
        if oS.alphas[i] > 0 and oS.alphas[i] < oS.C: oS.b = b1
        elif oS.alphas[j] > 0 and oS.alphas[j] < oS.C: oS.b = b2
        else: oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


#完整版SMO外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup):
    #初始化数据结构
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    #初始化当前迭代次数
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    #遍历整个数据集alpha也没有更新或者超过最大迭代次数，则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        if entireSet:
            #遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        else:
            #遍历非边界值，即alpha不为0或C
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        #遍历一次后改为非边界遍历
        if entireSet: entireSet = False
        #如果alpha没有更新，改为遍历整个数据集
        elif alphaPairsChanged == 0: entireSet = True
    return oS.b, oS.alphas


#利用核函数进行分类测试
def testKernel(kTup):
    #加载训练集
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    #根据训练集计算出b和alphas
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 100, kTup)
    dataMat = mat(dataArr); labelMat = mat(labelArr).transpose()
    #获得支持向量
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print '支持向量个数：%d' % shape(sVs)[0]
    m = shape(dataMat)[0]
    errorCount = 0
    for i in range(m):
        #计算各个数据的核
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        #根据支持向量的点，计算超平面，返回预测结果
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        #统计错误个数
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print '训练集错误率：%.2f%%' % float(100.0 * errorCount / m)
    #加载测试集
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    dataMat = mat(dataArr)
    m = shape(dataMat)[0]
    errorCount = 0
    #同训练集
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i,:], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print '测试集错误率：%.2f%%' % float(100.0 * errorCount / m)


if __name__ == '__main__':
    #线性SVM
    #testKernel(('lin', 0))
    #非线性SVM
    testKernel(('rbf', 1.4))