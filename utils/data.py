import torch
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.transforms import get_transforms
from utils.transforms import DiscoveryTargetTransform

import numpy as np
import os
from tqdm import tqdm

CLASS_NAMES=['Boat',  # 0
           'bulk cargo carrier',  # 1
           'Buoy',  # 2
           'container ship',  # 3
           'Ferry',  # 4
           'fishing boat',  # 5
           'flying bird',  # 6
           'general cargo ship',  # 7
           'Kayak',  # 8
           'ore carrier',  # 9
           'Other',  # 10
           'passenger ship',  # 11
           'Sail boat',  # 12
           'Speed boat',  # 13
           'vessel'  # 14
           ]
SHIP_ADVANCED_INDS=[14,0,1,2,3,4,5,7,8,9,10,11,12,13,6]

def get_datamodule(args, mode):
    if mode == "pretrain":
        if args.dataset == "ship":
            return PretrainShipDataModule(args)
        elif args.dataset == "ImageNet":
            return PretrainImageNetDataModule(args)
        else:
            return PretrainCIFARDataModule(args)
    elif mode == "discover":
        if args.dataset =="ship":
            return DiscoverShipDataModule(args)
        elif args.dataset == "ImageNet":
            return DiscoverImageNetDataModule(args)
        else:
            return DiscoverCIFARDataModule(args)
    elif mode=="pretrainfull":
        if args.dataset=="ship":
            return PretrainShipDataModule_full(args)


class PretrainCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)#返回属性
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )
        train_indices_lab = np.where(
            np.isin(np.array(self.train_dataset.targets), labeled_classes)
        )[0]
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab)

        # val datasets
        self.val_dataset = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        val_indices_lab = np.where(np.isin(np.array(self.val_dataset.targets), labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2
        )


class PretrainShipDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        #self.dataset_class = getattr(torchvision.datasets, 'CIFAR10')
        self.dataset= torchvision.datasets.ImageFolder(self.data_dir)#transform=transforms.ToTensor()
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_train_supervised = get_transforms("supervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.class_names = CLASS_NAMES
        #self.class_index = list(dict(list(zip(range(len(self.class_names)), self.class_names))).keys())


    def prepare_data(self):
        pass
        # self.dataset_class(os.path.join(self.data_dir,'..'), train=True, download=self.download)
        # self.dataset_class(os.path.join(self.data_dir,'..'), train=False, download=self.download)

    def setup(self, stage=None,eval_falg=False):
        from copy import copy
        #self.train_dataset1 = self.dataset_class(os.path.join(self.data_dir,'..'), train=True, transform=self.transform_train)
        assert len(self.class_names)>self.num_labeled_classes

        #labeled_classes = class_index#
        labeled_classes=np.arange(self.num_labeled_classes)
        #labeled_classes=self.labeled_advanced_inds[np.arange(self.num_labeled_classes)]

        N = len(self.dataset)
        train_size = int(N * 0.8)
        val_size = N - train_size

        trainDataset, valDataset = torch.utils.data.random_split(self.dataset,[train_size, val_size])
        trainDataset.dataset=copy(self.dataset)
        # train dataset
        self.train_dataset = trainDataset
        if eval_falg:
            self.train_dataset.dataset.transform = self.transform_val
        else:
            self.train_dataset.dataset.transform = self.transform_train
        subTarget=np.array(self.train_dataset.dataset.targets)[np.array(self.train_dataset.indices)]
        train_indices_lab = np.where(
            np.isin(subTarget, labeled_classes)
        )[0]
        # train_indices_unlab = np.where(
        #     np.isin(subTarget, range(self.num_labeled_classes,self.num_labeled_classes+self.num_unlabeled_classes))
        # )[0]
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab)
        # self.train_dataset.dataset.in
        # self.train_dataset.dataset.dataset[i2[i1[0]]]
        # val datasets
        self.val_dataset = valDataset
        self.val_dataset.dataset.transform = self.transform_val

        subTarget = np.array(self.val_dataset.dataset.targets)[np.array(self.val_dataset.indices)]
        val_indices_lab = np.where(np.isin(subTarget, labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def setup_test(self):
        from copy import copy
        #assert len(self.class_names)>self.num_labeled_classes

        #labeled_classes = class_index#
        labeled_classes=np.arange(self.num_labeled_classes)
        #labeled_classes=self.labeled_advanced_inds[np.arange(self.num_labeled_classes)]

        N = len(self.dataset)
        train_size = int(N * 0.8)
        val_size = N - train_size

        trainDataset, valDataset = torch.utils.data.random_split(self.dataset,[train_size, val_size])
        trainDataset.dataset=copy(self.dataset)
        # train dataset
        self.train_dataset = trainDataset
        # if eval_falg:
        #     self.train_dataset.dataset.transform = self.transform_train_supervised
        # else:
        # self.train_dataset.dataset.transform = self.transform_train
        self.train_dataset.dataset.transform = self.transform_val  # self.transform_train

        subTarget=np.array(self.train_dataset.dataset.targets)[np.array(self.train_dataset.indices)]
        train_indices_lab = np.where(
            np.isin(subTarget, labeled_classes)
        )[0]
        train_indices_unlab = np.where(
            np.isin(subTarget, range(self.num_labeled_classes,self.num_labeled_classes+self.num_unlabeled_classes))
        )[0]
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_indices_lab)
        i1=self.train_dataset.indices
        i2=self.train_dataset.dataset.indices

        target=np.array(self.train_dataset.dataset.dataset.targets)
        subT=target[np.array(i2)[np.array(i1)]]
        # i=3
        # print(self.train_dataset[i])
        # print(self.train_dataset.dataset[i1[i]])
        # print(self.train_dataset.dataset.dataset[i2[i1[i]]])
        # self.train_dataset.dataset.dataset[i2[i1[0]]]
        # val datasets
        self.val_dataset = valDataset
        self.val_dataset.dataset.transform = self.transform_val
        # getattr(self.train_dataset.dataset,'dataset')
        subTarget = np.array(self.val_dataset.dataset.targets)[np.array(self.val_dataset.indices)]
        val_indices_lab = np.where(np.isin(subTarget, labeled_classes))[0]
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, val_indices_lab)

    def setup_eval(self, eval_falg=False):
        from copy import copy

        N = len(self.dataset)
        train_size = int(N * 0.8)
        val_size = N - train_size

        trainDataset, valDataset = torch.utils.data.random_split(self.dataset,[train_size, val_size])
        trainDataset.dataset=copy(self.dataset)
        # train dataset
        self.train_dataset = trainDataset
        #subT=np.array(self.train_dataset.dataset.targets)[np.array(self.train_dataset.indices)]
        if eval_falg:
            self.train_dataset.dataset.transform = self.transform_train_supervised
        else:
            self.train_dataset.dataset.transform = self.transform_train

        self.val_dataset = valDataset
        self.val_dataset.dataset.transform = self.transform_val

    def make_weights_for_balanced_classes(self,dataset, nclasses):
        import pickle as pkl
        count = [0] * nclasses
        weight = [0] * len(dataset)
        val_list = [0] * len(dataset)
        cache_pkl=os.path.join(os.path.dirname(self.data_dir), 'cache')

        cache_basename=os.path.basename(self.data_dir)
        cache_pkl_file = os.path.join(cache_pkl, cache_basename+'%d_%d.pkl'%(self.num_labeled_classes,self.num_unlabeled_classes))
        if not os.path.exists(cache_pkl):
            os.mkdir(cache_pkl)
        if os.path.exists(cache_pkl_file):
            with open(cache_pkl_file,'rb') as f:
                weight_data=pkl.load(f)
                count=weight_data['count']
                val_list = weight_data['val_list']
            #(os.path.join(cache_pkl, cache_basename + '.pkl'))
        else:
            for idx, item in enumerate(tqdm(dataset)):
                # label=dataset[i][1]
                count[item[1]] += 1
                val_list[idx] = item[1]
            with open(cache_pkl_file,'wb') as f:
                pkl.dump({'count':count,'val_list':val_list},f)

        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            try:
                weight_per_class[i] = N / float(count[i])
            except ZeroDivisionError as e:
                weight_per_class[i]=1.0
                print(e)

        for idx, val in enumerate(tqdm(val_list)):
            weight[idx] = weight_per_class[val_list[idx]]

        return weight


    def train_dataloader(self,balanced=True):
        if balanced:
            weights = self.make_weights_for_balanced_classes(self.train_dataset, self.num_labeled_classes+self.num_unlabeled_classes)
            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                #shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2)
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                #sampler=sampler,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2
        )

class PretrainShipDataModule_full(PretrainShipDataModule):
    def __init__(self, args):
        super().__init__(args)
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        #self.dataset_class = getattr(torchvision.datasets, 'CIFAR10')
        self.dataset= torchvision.datasets.ImageFolder(self.data_dir)#transform=transforms.ToTensor()
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_train_supervised = get_transforms("supervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.class_names = CLASS_NAMES

    def setup(self, stage=None,eval_falg=False):
        from copy import copy

        N = len(self.dataset)
        train_size = int(N * 0.8)
        val_size = N - train_size

        trainDataset, valDataset = torch.utils.data.random_split(self.dataset,[train_size, val_size])
        trainDataset.dataset=copy(self.dataset)
        # train dataset
        self.train_dataset = trainDataset
        if eval_falg:
            self.train_dataset.dataset.transform = self.transform_val
        else:
            self.train_dataset.dataset.transform = self.transform_train

        self.val_dataset = valDataset
        self.val_dataset.dataset.transform = self.transform_val

    def train_dataloader(self,balanced=False):
        if balanced:
            weights = self.make_weights_for_balanced_classes(self.train_dataset, self.num_labeled_classes)
            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                #shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2)
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                #sampler=sampler,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2
        )
class DiscoverCIFARDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset_class = getattr(torchvision.datasets, args.dataset)
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)

    def prepare_data(self):
        self.dataset_class(self.data_dir, train=True, download=self.download)
        self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )

        # train dataset
        self.train_dataset = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_train
        )

        # val datasets
        val_dataset_train = self.dataset_class(
            self.data_dir, train=True, transform=self.transform_val
        )
        val_dataset_test = self.dataset_class(
            self.data_dir, train=False, transform=self.transform_val
        )
        # unlabeled classes, train set
        val_indices_unlab_train = np.where(
            np.isin(np.array(val_dataset_train.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)#根据索引划分子集
        # unlabeled classes, test set
        val_indices_unlab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)#根据索引划分子集
        # labeled classes, test set
        val_indices_lab_test = np.where(
            np.isin(np.array(val_dataset_test.targets), labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)#根据索引划分子集

        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=2
            )
            for dataset in self.val_datasets
        ]

class DiscoverShipDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.download = args.download
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_labeled_classes = args.num_labeled_classes
        self.num_unlabeled_classes = args.num_unlabeled_classes
        self.dataset = torchvision.datasets.ImageFolder(self.data_dir)
        #self.dataset_class = getattr(torchvision.datasets, 'CIFAR10')
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)
        self.class_names = CLASS_NAMES
        #self.class_index = list(dict(list(zip(range(len(self.class_names)), self.class_names))).keys())
        #self.labeled_advanced_inds=np.array(SHIP_ADVANCED_INDS)

    def prepare_data(self):
        pass
        # self.dataset_class(self.data_dir, train=True, download=self.download)
        # self.dataset_class(self.data_dir, train=False, download=self.download)

    def setup(self, stage=None):
        from copy import copy

        assert len(self.class_names) > self.num_labeled_classes

        labeled_classes = range(self.num_labeled_classes)
        unlabeled_classes = range(
            self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        )
        # labeled_classes = self.labeled_advanced_inds[np.arange(self.num_labeled_classes)]
        # unlabeled_classes = self.labeled_advanced_inds[np.arange(
        #     self.num_labeled_classes, self.num_labeled_classes + self.num_unlabeled_classes
        # )]

        N = len(self.dataset)
        train_size = int(N * 0.8)
        val_size = int(N * 0.1)
        test_size = N - train_size - val_size
        trainDataset, valDataset, testDataset = torch.utils.data.random_split(self.dataset,
                                                                              [train_size, val_size, test_size])
        trainDataset.dataset = copy(self.dataset)
        valDataset.dataset = copy(self.dataset)
        # train dataset
        #self.dataset_class
        self.train_dataset = trainDataset
        self.train_dataset.dataset.transform=self.transform_train

        # val datasets
        val_dataset_train = valDataset
        val_dataset_train.dataset.transform=self.transform_val

        val_dataset_test = testDataset
        val_dataset_test.dataset.transform=self.transform_val

        # unlabeled classes, train set
        subTarget = np.array(val_dataset_train.dataset.targets)[np.array(val_dataset_train.indices)]
        val_indices_unlab_train = np.where(
            np.isin(subTarget, unlabeled_classes)
        )[0]

        val_subset_unlab_train = torch.utils.data.Subset(val_dataset_train, val_indices_unlab_train)#根据索引划分子集
        # unlabeled classes, test set
        subTarget = np.array(val_dataset_test.dataset.targets)[np.array(val_dataset_test.indices)]
        val_indices_unlab_test = np.where(
            np.isin(subTarget, unlabeled_classes)
        )[0]
        val_subset_unlab_test = torch.utils.data.Subset(val_dataset_test, val_indices_unlab_test)#根据索引划分子集
        # labeled classes, test set
        #subTarget = np.array(val_dataset_test.dataset.targets)[np.array(val_dataset_test.indices)]
        val_indices_lab_test = np.where(
            np.isin(subTarget, labeled_classes)
        )[0]
        val_subset_lab_test = torch.utils.data.Subset(val_dataset_test, val_indices_lab_test)#根据索引划分子集

        self.val_datasets = [val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def make_weights_for_balanced_classes(self,dataset, nclasses):
        count = [0] * nclasses
        weight = [0] * len(dataset)
        val_list = [0] * len(dataset)
        # tmp_dataset=dataset
        # while isinstance(tmp_dataset,torch.utils.data.dataset.Subset):
        #     tmp_dataset=tmp_dataset.dataset
        # td=torch.utils.data.Subset(dataset,[0,1,2])
        for idx, item in enumerate(tqdm(dataset)):
            # label=dataset[i][1]
            count[item[1]] += 1
            val_list[idx] = item[1]
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])

        for idx, val in enumerate(tqdm(val_list)):
            #weight[idx] = weight_per_class[val[1]]
            weight[idx] = weight_per_class[val_list[idx]]
        return weight

    def train_dataloader(self,balanced=False):
        # from tqdm import tqdm
        # coll=[0]*(self.num_labeled_classes+self.num_unlabeled_classes)
        # for item in tqdm(self.train_dataset):
        #     coll[item[1]]+=1
        if balanced:
            weights = self.make_weights_for_balanced_classes(self.train_dataset, self.num_labeled_classes+self.num_unlabeled_classes)
            weights = torch.DoubleTensor(weights)
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                #shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2)
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                #sampler=sampler,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2)


    def val_dataloader(self):
        dataloader_list=[]
        for dataset in self.val_datasets:
            dataloader_list.append(
                DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=2))

        return dataloader_list

IMAGENET_CLASSES_118 = [
    "n01498041",
    "n01537544",
    "n01580077",
    "n01592084",
    "n01632777",
    "n01644373",
    "n01665541",
    "n01675722",
    "n01688243",
    "n01729977",
    "n01775062",
    "n01818515",
    "n01843383",
    "n01883070",
    "n01950731",
    "n02002724",
    "n02013706",
    "n02092339",
    "n02093256",
    "n02095314",
    "n02097130",
    "n02097298",
    "n02098413",
    "n02101388",
    "n02106382",
    "n02108089",
    "n02110063",
    "n02111129",
    "n02111500",
    "n02112350",
    "n02115913",
    "n02117135",
    "n02120505",
    "n02123045",
    "n02125311",
    "n02134084",
    "n02167151",
    "n02190166",
    "n02206856",
    "n02231487",
    "n02256656",
    "n02398521",
    "n02480855",
    "n02481823",
    "n02490219",
    "n02607072",
    "n02666196",
    "n02672831",
    "n02704792",
    "n02708093",
    "n02814533",
    "n02817516",
    "n02840245",
    "n02843684",
    "n02870880",
    "n02877765",
    "n02966193",
    "n03016953",
    "n03017168",
    "n03026506",
    "n03047690",
    "n03095699",
    "n03134739",
    "n03179701",
    "n03255030",
    "n03388183",
    "n03394916",
    "n03424325",
    "n03467068",
    "n03476684",
    "n03483316",
    "n03627232",
    "n03658185",
    "n03710193",
    "n03721384",
    "n03733131",
    "n03785016",
    "n03786901",
    "n03792972",
    "n03794056",
    "n03832673",
    "n03843555",
    "n03877472",
    "n03899768",
    "n03930313",
    "n03935335",
    "n03954731",
    "n03995372",
    "n04004767",
    "n04037443",
    "n04065272",
    "n04069434",
    "n04090263",
    "n04118538",
    "n04120489",
    "n04141975",
    "n04152593",
    "n04154565",
    "n04204347",
    "n04208210",
    "n04209133",
    "n04258138",
    "n04311004",
    "n04326547",
    "n04367480",
    "n04447861",
    "n04483307",
    "n04522168",
    "n04548280",
    "n04554684",
    "n04597913",
    "n04612504",
    "n07695742",
    "n07697313",
    "n07697537",
    "n07716906",
    "n12998815",
    "n13133613",
]

IMAGENET_CLASSES_30 = {
    "A": [
        "n01580077",
        "n01688243",
        "n01883070",
        "n02092339",
        "n02095314",
        "n02098413",
        "n02108089",
        "n02120505",
        "n02123045",
        "n02256656",
        "n02607072",
        "n02814533",
        "n02840245",
        "n02843684",
        "n02877765",
        "n03179701",
        "n03424325",
        "n03483316",
        "n03627232",
        "n03658185",
        "n03785016",
        "n03794056",
        "n03899768",
        "n04037443",
        "n04069434",
        "n04118538",
        "n04154565",
        "n04311004",
        "n04522168",
        "n07695742",
    ],
    "B": [
        "n01883070",
        "n02013706",
        "n02093256",
        "n02097130",
        "n02101388",
        "n02106382",
        "n02112350",
        "n02167151",
        "n02490219",
        "n02814533",
        "n02843684",
        "n02870880",
        "n03017168",
        "n03047690",
        "n03134739",
        "n03394916",
        "n03424325",
        "n03483316",
        "n03658185",
        "n03721384",
        "n03733131",
        "n03786901",
        "n03843555",
        "n04120489",
        "n04152593",
        "n04208210",
        "n04258138",
        "n04522168",
        "n04554684",
        "n12998815",
    ],
    "C": [
        "n01580077",
        "n01592084",
        "n01632777",
        "n01775062",
        "n01818515",
        "n02097130",
        "n02097298",
        "n02098413",
        "n02111500",
        "n02115913",
        "n02117135",
        "n02398521",
        "n02480855",
        "n02817516",
        "n02843684",
        "n02877765",
        "n02966193",
        "n03095699",
        "n03394916",
        "n03424325",
        "n03710193",
        "n03733131",
        "n03785016",
        "n03995372",
        "n04090263",
        "n04120489",
        "n04326547",
        "n04522168",
        "n07697537",
        "n07716906",
    ],
}


class DiscoveryDataset:
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __len__(self):
        return max([len(self.labeled_dataset), len(self.unlabeled_dataset)])

    def __getitem__(self, index):
        labeled_index = index % len(self.labeled_dataset)
        labeled_data = self.labeled_dataset[labeled_index]
        unlabeled_index = index % len(self.unlabeled_dataset)
        unlabeled_data = self.unlabeled_dataset[unlabeled_index]
        return (*labeled_data, *unlabeled_data)


class DiscoverImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.imagenet_split = args.imagenet_split
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # split classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]
        unlabeled_classes = IMAGENET_CLASSES_30[self.imagenet_split]
        unlabeled_classes.sort()
        unlabeled_class_idxs = [mapping[c] for c in unlabeled_classes]

        # target transform
        all_classes = labeled_classes + unlabeled_classes
        target_transform = DiscoveryTargetTransform(
            {mapping[c]: i for i, c in enumerate(all_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        labeled_subset = torch.utils.data.Subset(train_dataset, labeled_idxs)
        unlabeled_idxs = np.where(np.isin(targets, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset = torch.utils.data.Subset(train_dataset, unlabeled_idxs)
        self.train_dataset = DiscoveryDataset(labeled_subset, unlabeled_subset)

        # val datasets
        val_dataset_train = self.dataset_class(
            train_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        val_dataset_test = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets_train = np.array([img[1] for img in val_dataset_train.imgs])
        targets_test = np.array([img[1] for img in val_dataset_test.imgs])
        # unlabeled classes, train set
        unlabeled_idxs = np.where(np.isin(targets_train, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_train = torch.utils.data.Subset(val_dataset_train, unlabeled_idxs)
        # unlabeled classes, test set
        unlabeled_idxs = np.where(np.isin(targets_test, np.array(unlabeled_class_idxs)))[0]
        unlabeled_subset_test = torch.utils.data.Subset(val_dataset_test, unlabeled_idxs)
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets_test, np.array(labeled_class_idxs)))[0]
        labeled_subset_test = torch.utils.data.Subset(val_dataset_test, labeled_idxs)

        self.val_datasets = [unlabeled_subset_train, unlabeled_subset_test, labeled_subset_test]

    @property
    def dataloader_mapping(self):
        return {0: "unlab/train", 1: "unlab/test", 2: "lab/test"}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size // 2,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=False,
                prefetch_factor=2
            )
            for dataset in self.val_datasets
        ]


class PretrainImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_class = torchvision.datasets.ImageFolder
        self.transform_train = get_transforms("unsupervised", args.dataset, args.num_views)
        self.transform_val = get_transforms("eval", args.dataset, args.num_views)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_data_dir = os.path.join(self.data_dir, "train")
        val_data_dir = os.path.join(self.data_dir, "val")

        # train dataset
        train_dataset = self.dataset_class(train_data_dir, transform=self.transform_train)

        # find labeled classes
        mapping = {c[:9]: i for c, i in train_dataset.class_to_idx.items()}
        labeled_classes = list(set(mapping.keys()) - set(IMAGENET_CLASSES_118))
        labeled_classes.sort()
        labeled_class_idxs = [mapping[c] for c in labeled_classes]

        # target transform
        target_transform = DiscoveryTargetTransform(
            {mapping[c]: i for i, c in enumerate(labeled_classes)}
        )
        train_dataset.target_transform = target_transform

        # train set
        targets = np.array([img[1] for img in train_dataset.imgs])
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.train_dataset = torch.utils.data.Subset(train_dataset, labeled_idxs)

        # val datasets
        val_dataset = self.dataset_class(
            val_data_dir, transform=self.transform_val, target_transform=target_transform
        )
        targets = np.array([img[1] for img in val_dataset.imgs])
        # labeled classes, test set
        labeled_idxs = np.where(np.isin(targets, np.array(labeled_class_idxs)))[0]
        self.val_dataset = torch.utils.data.Subset(val_dataset, labeled_idxs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2
        )
