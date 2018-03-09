#!/usr/bin/env python
#-*-coding:utf-8-*-

from math import log
import operator
import pickle
import treePlotter


#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    #计算数据集的行数
    numEntries = len(dataSet)
    #保存每个标签出想次数的字典
    labelCounts = {}
    #对每组特征向量进行统计
    for featVec in dataSet:
        #提取标签信息
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        #标签次数计数
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        #计算key标签的概率
        prob = float(labelCounts[key]) / numEntries
        #利用公式计算香农熵
        shannonEnt -= prob * log(prob, 2)
    #返回香农熵
    return shannonEnt


#划分数据集
def splitDataSet(dataSet, axis, value):
    #创建返回的数据集列表
    retDataSet = []
    #遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])
            #将符合条件的向量添加到返回的数据集中
            retDataSet.append(reducedFeatVec)
    #返回划分后的数据集
    return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    #特征数量
    numFeatures = len(dataSet[0]) - 1
    #计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #信息增益
    bestInfoGain = 0.0
    #最优特征的索引值
    bestFeature = -1
    #遍历所有特征
    for i in range(numFeatures):
        #获取dataSet的第i个所有特征值
        featList = [example[i] for example in dataSet]
        #创建集合，得到第i个特征对应的所有唯一特征值
        uniqueVals = set(featList)
        #计算信息增益
        newEntropy = 0.0
        for value in uniqueVals:
            #得到划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            #计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            #根据公式计算香农熵
            newEntropy -= prob * calcShannonEnt(subDataSet)
        #计算信息增益
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            #更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            #记录信息增益最大的特征的索引值
            bestFeature = i
    #返回信息增益最大的特征的索引值
    return bestFeature


#统计classList中出现此处最多的元素(类标签)
def majorityCnt(classList):
    classCount = {}
    #统计classList中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #根据字典的值降序排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    #返回classList中出现次数最多的元素
    return sortedClassCount[0][0]


#创建决策树
def createTree(dataSet, labels):
    #取出分类标签
    classList = [example[-1] for example in dataSet]
    #如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #最优特征的标签
    bestFeatLabel = labels[bestFeat]
    #根据最优特征的标签生成树
    myTree = {bestFeatLabel:{}}
    #删除已经使用的特征标签
    del(labels[bestFeat])
    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    #去掉重复的属性值
    uniqueVals = set(featValues)
    #遍历特征，创建决策树
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


#使用决策树执行分类
def classify(inputTree, featLabels, testVec):
    #获取决策树的结点
    firstStr = inputTree.keys()[0]
    #获取下一个字典
    secondDict = inputTree[firstStr]
    #将标签字符串转换为索引
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
             #递归到下一个字典
            classLabel = classify(secondDict[key], featLabels, testVec)
        else:#到达叶子节点，返回当前节点的分类标签
            classLabel = secondDict[key]
    return classList


#决策树的存储
def storeTree(inputTree, filename):
    with open(filename, 'w') as fw:
        pickle.dump(inputTree, fw)


#读取决策树
def grabTree(filename):
    with open(filename) as fr:
        return pickle.load(fr)


#使用决策树预测隐形眼镜类型
if __name__ == '__main__':
    #加载并处理文件
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    #标签列表
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    #创建决策树
    lensesTree = createTree(lenses, lensesLabels)
    #打印树
    print lensesTree
    #显示树形图
    treePlotter.createPlot(lensesTree)
