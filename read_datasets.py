import pickle

#精细类别的序号与名称 序号:名称
fineLabelNameDict={}
#精细类别对应的粗糙类别 精细序号：粗糙序号-粗糙名称

fineLableToCoraseLabelDict={}

def unpickle(file):
    #import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# 给定路径添加数据
def Dealdata(meta, train):
    for fineLabel, coarseLabel in zip(train[b'fine_labels'], train[b'coarse_labels']):
        if fineLabel not in fineLabelNameDict.keys():
            fineLabelNameDict[fineLabel]=meta[b'fine_label_names'][fineLabel].decode('utf-8')
        if fineLabel not in fineLableToCoraseLabelDict.keys():
            fineLableToCoraseLabelDict[fineLabel]=str(coarseLabel)+"-"+meta[b'coarse_label_names'][coarseLabel].decode('utf-8')


metaPath = 'datasets/cifar-100-python/meta'
# 解压后train的路径
trainPath = 'datasets/cifar-100-python/train'

meta = unpickle(metaPath)
train = unpickle(trainPath)
Dealdata(meta, train)
cifar100_class_names=dict(enumerate(meta[b'fine_label_names']))
print(cifar100_class_names)
print(meta[b'fine_label_names'])
# print(len(fineLabelNameDict))
# print(len(fineLableToCoraseLabelDict))
# print(sorted(fineLabelNameDict.items(),key=lambda x:x[0]))
# print(sorted(fineLableToCoraseLabelDict.items(),key=lambda x:x[0]))
