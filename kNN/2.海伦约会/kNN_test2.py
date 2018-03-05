#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import operator


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


#从文本文件中解析数据
def file2matrix(filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOLines)
    #定义返回的numpy矩阵，numberOfLines行，3列
    returnMat = zeros((numberOfLines, 3))
    #返回的分类标签向量
    classLabelVector = []
    #行的索引值
    index = 0
    for line in arrayOLines:
        #strip(rm)，当rm为空时，默认删除空白符(包括'\n', '\r', '\t', ' ')
        line = line.strip()
        #使用split()将字符串根据'\t'分隔符进行切片
        listFromLine = line.split('\t')
        #将数据前三列提取出来，存放到returnMat的numpy矩阵中，即特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #得到对应的标签向量
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat, classLabelVector


#归一化数值
def autoNorm(dataSet):
    #获得数据集的最小和最大值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #数值范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = zeros(shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    #除以最大值和最小值的差，得到归一化的数据
    normDataSet = normDataSet / tile(ranges, (m, 1))
    #返回归一化数据结果，数据范围，最小值
    return normDataSet, ranges, minVals


#验证分类器
def datingClassTest():
    #取所有数据的百分之十
    hoRatio = 0.10
    #将返回的特征矩阵和分类向量分别存到datintDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获取normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类结果错误计数
    errorCount = 0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集，后m-numTestVecs个数据作为训练集
        classifyResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %s, the real answer is: %s" % (classifyResult, datingLabels[i])
        if (classifyResult != datingLabels[i]):
            errorCount += 1.0
    #输出分类结果错误率
    print "the total error rate is: %f" % (100.0 * errorCount / numTestVecs) + '%'


#构建完整可用系统
def classifyPerson():
    #三维特征用户输入
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    #将返回的特征矩阵和分类向量分别存到datintDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    #数据归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = array([ffMiles, precentTats, iceCream])
    #返回分类结果
    classifyResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    #打印结果
    print "You will probably %s this person" % classifyResult


if __name__ == '__main__':
    datingClassTest()  #测试结果：k=3时错误率为5%
    classifyPerson()
