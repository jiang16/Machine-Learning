#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


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


if __name__ == '__main__':
    group, labels = createDataSet()
    #测试向量
    test = [0, 0]
    #输出分类结果
    print classify(test, group, labels, 3)
