import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import random
import colorsys
from collections import Counter
def t_sne_projection(x,y=None,dims=2):
    sns.set(color_codes=True)
    #sns.set(rc={'figure.figsize': (11.7, 8.27)})
    #palette = sns.color_palette("bright", 80)
    tsne = TSNE(n_components=dims)

    x_embedded=tsne.fit_transform(x)  # 进行数据降维,降成两维
    # a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    # plt.figure()
    # sns.scatterplot(x_embedded[:, 0], x_embedded[:, 1], hue=y, legend='full', )#palette=palette
    #plt.show()
    return x_embedded#y

def dataset_visualization(data,dataset_name='CIFAR10'):
    mean, std = {
        "CIFAR10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "CIFAR100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "ImageNet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }[dataset_name]
    data1=(data * torch.reshape(torch.Tensor(std), (3, 1, 1))+torch.reshape(torch.Tensor(mean),(3,1,1)))*255
    data1=np.transpose(data1.numpy(), (1, 2, 0))

    plt.imshow(data1.astype(np.uint8))
    plt.show()

def drawDiffClass(x,y,colors):
    for i in set(y):
        indices = np.where(
            np.isin(np.array(y), i)
        )[0]
        plt.scatter(x[indices, 0], x[indices, 1], c=colors[i], marker='o',
                    edgecolors='k')  # sns.color_palette(palettes[0])


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
def norm1(x,axis=None):
    return x / x.sum(axis=axis, keepdims=True)
