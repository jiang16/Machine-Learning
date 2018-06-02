#!/usr/bin/env python
#-*-coding:utf-8-*-


#创建一个用于测试的简单数据集
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


#构建初始候选项集的列表C1，即所有候选项集只包含一个元素
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #对C1中每个项构建一个不变集合
    return map(frozenset, C1)


#计算Ck中的项集在数据集D中的支持度，返回满足最小支持度的项集集合Lk，和所有项集支持度信息的字典
def scanD(D, Ck, minSupport):
    ssCnt = {}
    #对每一条交易信息tid
    for tid in D:
        #对每一个候选项集can，检查是否是tid的一部分，即是否得到支持
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can] = 1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}   
    for key in ssCnt:
        #计算每个项集的支持度
        support = ssCnt[key] / numItems
        #将满足最小支持度的项集加入retList
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


#由初始候选项集的集合Lk生成新的候选项集Ck，k表示生成的新项集中所含元素个数
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            #前k-2个项相同时，将两个集合合并(确保遍历列表的次数最少)
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


#Aprior算法
def apriori(dataSet, minSupport = 0.5):
    #构建初始候选项集C1
    C1 = createC1(dataSet)
    #将dataSet集合化，以满足scanD的格式要求
    D = map(set, dataSet)
    #构建初始的频繁项集
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    #最初的L1中每个项集只含有一个元素，新生成的项集应该含有两个元素
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        #将新的项集支持度数据加入原来的总支持度字典中
        supportData.update(supK)
        #将符合最小支持度的项集加入L
        L.append(Lk)
        #新生成的项集中元素个数应不断增加
        k += 1
    #返回所有满足条件的频繁项集列表和所有频繁项集的支持度信息
    return L, supportData


#计算规则的可信度，返回满足最小可信度的规则
def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
    prunedH = []
    for conseq in H:
        #计算规则可信度
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            #打印规则P --> H
            print freqSet - conseq, '-->', conseq, 'conf:', conf
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


#对频繁项集中元素超过2的项集进行合并
def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
    m = len(H[0])
    #检查频繁项集是否可以移除大小为m+1的子集
    if len(freqSet) > (m + 1):
        #尝试进一步合并
        Hmp1 = aprioriGen(H, m + 1)
        #创建Hm+1条新候选规则
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        #如果不止一条规则满足要求，进一步递归合并
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)


#根据频繁项集和最小可信度生成关联规则
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    #只获取有两个或更多元素的集合
    for i in range(1, len(L)):
        #对每一个频繁项集的集合
        for freqSet in L[i]:            
            H1 = [frozenset([item]) for item in freqSet]
            #如果频繁项集中的元素个数大于2,需要进一步合并
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            #频繁项集中只有两个元素，直接调用函数计算可信度
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


if __name__ == '__main__':
    #导入数据集
    myDat = loadDataSet()
    #生成频繁项集
    L, suppData = apriori(myDat, 0)
    #从频繁项集中挖掘关联规则
    rules = generateRules(L, suppData, 0)
    print 'rules:', rules

    # 发现毒蘑菇的相似特征测试
    # mushDataSet = [line.split() for line in open('mushroom.dat').readlines()]
    # L, suppData = apriori(mushDataSet, 0.3)
    # 在结果中搜索包含有毒特征值2的频繁项集
    # for item in L[3]:
    #     if item.intersection('2'): print item