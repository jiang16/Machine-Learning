#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *

#FP树的类定义
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    #用于将树以文本形式显示
    def disp(self, ind = 1):
        print ' ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)


#FP树构建函数
def createTree(dataSet, minsup = 1):
    #头指针表
    headerTable = {}
    #第一次遍历数据集，统计每个元素项出现的频数
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    #移除不满足最小支持度的元素项
    for k in headerTable.keys():
        if headerTable[k] < minsup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    #如果没有元素项满足要求，则退出
    if len(freqItemSet) == 0: return None, None
    #头指针表保存技术值和指向每种类型第一个元素项的指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    #创建只包含空集合的根节点
    retTree = treeNode('Null set', 1, None)
    #第二次遍历数据集
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            #只考虑频繁项
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            #根据全局频率对每个事务中的元素进行排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            #使用排序后的频率项集对树进行填充
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


#让FP树生长
def updateTree(items, inTree, headerTable, count):
    #测试事务中的第一个元素项是否作为子节点存在
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        #创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        #头指针表更新以指向新的节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    #对剩下的元素项迭代调用updateTree函数
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


#确保节点链接指向树中该元素项的每一个实例
def updateHeader(nodeToTest, targetNode):
    #从头指针表的nodeLink开始，一直沿着nodeLink直到到达链表末尾
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


#简单数据集
def loadSimpleDat():
    simpleDat = [
        ['r', 'z', 'h', 'j', 'p'],
        ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
        ['z'],
        ['r', 'x', 'n', 'o', 's'],
        ['y', 'r', 'x', 'z', 'q', 't', 'p'],
        ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
    ]
    return simpleDat


#将数据集从列表转换为字典
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        #若没有相同事项，则为1；若有相同事项，则加1  
        retDict[frozenset(trans)] = retDict.get(frozenset(trans), 0) + 1
    return retDict


#上溯FP树，并收集所有遇到的元素项的名称
def ascendTree(leafNode, prefixPath):
    #迭代上溯整颗FP树
    while leafNode.parent != None:
        #收集所有遇到的元素项的名称
        prefixPath.append(leafNode.name)
        leafNode = leafNode.parent


#遍历链表直到到达结尾
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        #每遇到一个元素项都调用ascendTree()来上溯FP树
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            #将列表添加到条件模式基字典中
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    #返回条件模式基字典
    return condPats


#递归查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemSet):
    #对头指针表中的元素项按照出现频率进行排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #将每一个频繁项添加到频繁项集列表freqItemSet
        freqItemSet.append(newFreqSet)
        #创建条件基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #构建条件树
        myCondTree, myHead = createTree(condPattBases, minSup)
        #如果树中有元素项，递归调用
        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemSet)


if __name__ == '__main__':
    # initSet = createInitSet(loadSimpleDat())
    # myFPTree, myHeaderTab = createTree(initSet, 3)
    # print myFPTree.disp()
    # freqItems = []
    # mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
    # print freqItems
    #寻找那些至少被十万人浏览过的新闻报道
    parsedDat = [line.strip().split(' ') for line in open('kosarak.dat').readlines()]
    initSet = createInitSet(parsedDat)
    myFPTree, myHeaderTab = createTree(initSet, 100000)
    myFreqList = []
    mineTree(myFPTree, myHeaderTab, 100000, set([]), myFreqList)
    print len(myFreqList)
    print myFreqList
