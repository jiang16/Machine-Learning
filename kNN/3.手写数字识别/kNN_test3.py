#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import operator
from os import listdir


def classify(inX, dataSet, labels, k):
    #numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    #在列向量方向上重复inX共1次（横向），行向量方向上重复inX共dataSetSize次（纵向）
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征值相减后平方
    sqDiffMat = diffMat ** 2
    #sum()所有元素相加，sum(0)列相减，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis = 1)
    #开方，计算出欧式距离
    distances = sqDistances ** 0.5
    #返回distances中元素从小到大排序后的下标
    sortedDistIndicies = distances.argsort()
    #记录类别的字典
    classCount = {}
    for i in range(k):
        #循环取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key, default=None),字典的get()方法，返回指定键的值，如果值不在字典中返回默认值
        #计算类别出现的次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #key=operator.itemgetter(0)根据字典的键进行排序
    #key=operator.itemgetter(1)根据字典的值进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    #返回次数最多的类别，即所要分类的类别
    return sortedClassCount[0][0]


#将图像转换为测试向量
def img2vector(filename):
    #创建1x1024的向量
    returnVect = zeros((1, 1024))
    #打开文件
    fr = open(filename)
    #按行读取
    for i in range(32):
        #读一行数据
        lineStr = fr.readline()
        #将每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0, 32*i+j] = lineStr[j]
    return returnVect


#使用kNN算法识别手写数字
def handwriteClassTest():
    #测试集的标签
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir('trainingDigits')
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = zeros((m, 1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
	    #获得文件的名字
	    fileNameStr = trainingFileList[i]
	    #获得分类的数字
	    classNumber = int(fileNameStr.split('_')[0])
	    #将获得的类别添加到hwLabels中
	    hwLabels.append(classNumber)
	    #将每一个文件的1x1024数据存储到trainingMat矩阵中
	    trainingMat[i,:] = img2vector('trainingDigits/%s' % (fileNameStr))
    #返回testDigits目录下的文件名
    testFileList = listdir('testDigits')
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #获得分类的数字
        classNumber = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量，用于训练
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        #获得预测结果
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumber)
        if (classifierResult != classNumber):
            errorCount += 1.0
    print "the total number of errors is: %d" % errorCount
    print "the total error rate is: %f" % (100.0 * errorCount / float(mTest)) + '%'


if __name__ == '__main__':
    handwriteClassTest()
