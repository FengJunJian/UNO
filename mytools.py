import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def t_sne_projection(x,y=None,dims=2):
    # x=np.empty((0,6),np.float)
    # y=np.empty((0),np.int)
    # for i in range(5):
    #     x0=np.random.normal(i,0.8,[100,6])
    #     y0=np.ones(100,dtype=np.int)*i
    #     x=np.concatenate([x,x0],0)
    #     y=np.concatenate([y,y0],0)

    #sns.set(rc={'figure.figsize': (11.7, 8.27)})
    #palette = sns.color_palette("bright", 80)
    tsne = TSNE(n_components=dims)

    x_embedded=tsne.fit_transform(x)  # 进行数据降维,降成两维
    # a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    sns.scatterplot(x_embedded[:, 0], x_embedded[:, 1], hue=y, legend='full', )#palette=palette
    plt.show()

def dataset_visualization(data,dataset_name='CIFAR10'):
    #import PIL

    mean, std = {
        "CIFAR10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "CIFAR100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "ImageNet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }[dataset_name]
    data1=(data * torch.reshape(torch.Tensor(std), (3, 1, 1))+torch.reshape(torch.Tensor(mean),(3,1,1)))*255
    data1=np.transpose(data1.numpy(), (1, 2, 0))

    plt.imshow(data1.astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    begin=ord('a')
    class_name=[chr(begin+i) for i in range(26)]
    data=np.load('proj_numpy_test.npz')
    class_range=[0,1,2,15]#range(15)
    indices_data = np.where(
        np.isin(np.array(data['y']), class_range)
    )[0]
    target=list(map(lambda x:class_name[x],data['y'][indices_data]))
    t_sne_projection(data['x'][indices_data], y=target, dims=2)

