import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics
from utils.sinkhorn_knopp import SinkhornKnopp

import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
import os
from mytools import t_sne_projection,dataset_visualization
from collections import Counter


parser = ArgumentParser()
parser.add_argument("--dataset", default="ship", type=str, help="dataset")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")
parser.add_argument("--data_dir", default="datasets/Classification_advanced", type=str, help="data directory")#Classification
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=16, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=4, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--num_views", default=2, type=int, help="number of views")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default='ship_13_1', type=str)
parser.add_argument("--project", default="UNO", type=str, help="wandb project")
parser.add_argument("--entity", default='chfjj', type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=13, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=2, type=int, help="number of unlab classes")

parser.add_argument("--pretrained", default='checkpoints/pretrain-resnet18-ship-ship_13_1.cp',type=str, help="pretrained checkpoint path")#checkpoints/epoch=29-step=8039_ship13_2.ckpt
parser.add_argument("--checkpoints", default='ship_13_1/final_modelship.pth',type=str, help="checkpoint path")

# class_names=[b'apple', b'aquarium_fish', b'baby', b'bear', b'beaver', b'bed', b'bee', b'beetle', b'bicycle', b'bottle', b'bowl', b'boy', b'bridge', b'bus', b'butterfly', b'camel', b'can', b'castle', b'caterpillar', b'cattle', b'chair', b'chimpanzee', b'clock', b'cloud', b'cockroach', b'couch', b'crab', b'crocodile', b'cup', b'dinosaur', b'dolphin', b'elephant', b'flatfish', b'forest', b'fox', b'girl', b'hamster', b'house', b'kangaroo', b'keyboard', b'lamp', b'lawn_mower', b'leopard', b'lion', b'lizard', b'lobster', b'man', b'maple_tree', b'motorcycle', b'mountain', b'mouse', b'mushroom', b'oak_tree', b'orange', b'orchid', b'otter', b'palm_tree', b'pear', b'pickup_truck', b'pine_tree', b'plain', b'plate', b'poppy', b'porcupine', b'possum', b'rabbit', b'raccoon', b'ray', b'road', b'rocket', b'rose', b'sea', b'seal', b'shark', b'shrew', b'skunk', b'skyscraper', b'snail', b'snake', b'spider', b'squirrel', b'streetcar', b'sunflower', b'sweet_pepper', b'table', b'tank', b'telephone', b'television', b'tiger', b'tractor', b'train', b'trout', b'tulip', b'turtle', b'wardrobe', b'whale', b'willow_tree', b'wolf', b'woman', b'worm']

class_names=['Boat',#0
             'bulk cargo carrier',#1
             'Buoy',#2
             'container ship',#3
             'Ferry',#4
             'fishing boat',#5
             'flying bird',#6
             'general cargo ship',#7
             'Kayak',#8
             'ore carrier',#9
             'Other',#10
             'passenger ship',#11
             'Sail boat',#12
             'Speed boat',#13
             'vessel'#14
             ]
class Pretrainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset or "ship" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            num_heads=None,
        )

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            self.model.load_state_dict(state_dict, strict=False)

        # metrics
        self.accuracy = Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(images)

        # supervised loss
        loss_supervised = torch.stack(
            [F.cross_entropy(o / self.hparams.temperature, labels) for o in outputs["logits_lab"]]
        ).mean()

        # log
        results = {
            "loss_supervised": loss_supervised,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)

        # reweight loss
        return loss_supervised

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # forward
        logits = self.model(images)["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, labels)
        acc = self.accuracy(preds, labels)

        # log
        results = {
            "val/loss_supervised": loss_supervised,
            "val/acc": acc,
        }
        self.log_dict(results, on_step=False, on_epoch=True)
        return results

class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset or "ship" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
        )

        if self.hparams.pretrained:
            state_dict = torch.load(self.hparams.pretrained)
            state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
            self.model.load_state_dict(state_dict, strict=False)

        # Sinkorn-Knopp #########################################
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )

        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=1)
        return -torch.mean(torch.sum(targets * preds, dim=1))

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_views):
            for other_view in np.delete(range(self.hparams.num_views), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_views * (self.hparams.num_views - 1))

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        if self.hparams.dataset == "ImageNet":
            views_lab, labels_lab, views_unlab, labels_unlab = batch
            views = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
            labels = torch.cat([labels_lab, labels_unlab])
        else:
            views, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return views, labels, mask_lab

    def training_step(self, batch, idx):
        views, labels, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(views)

        # gather outputs
        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)

        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(self.hparams.num_views):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)

        # compute swapped prediction loss
        loss_cluster, loss_overcluster = [], []
        for h in range(self.hparams.num_heads):
            loss_cluster.append(self.swapped_prediction(logits, targets))
            loss_overcluster.append(self.swapped_prediction(logits_over, targets_over))

        # total loss
        loss_cluster = torch.stack(loss_cluster)
        loss_overcluster = torch.stack(loss_overcluster)
        loss = (loss_cluster + loss_overcluster).mean()

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # log
        results = {
            "loss": loss.detach(),
            "loss_cluster": loss_cluster.mean(),
            "loss_overcluster": loss_overcluster.mean(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dl_idx):
        images, labels = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self(images)

        if "unlab" in tag:  # use clustering head
            preds = outputs["logits_unlab"]
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            preds = outputs["logits_lab"]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)

    def validation_epoch_end(self, _):
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result)
                self.log(prefix_inc + "/acc", result_inc)

def main_discover(args):
    dm = get_datamodule(args, "discover")
    dm.setup()
    dataloader=dm.train_dataloader(False)
    valdataloaders=dm.val_dataloader()#[val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    model = Discoverer(**args.__dict__)

    state_dict = torch.load(args.pretrained)#  epoch=29-step=5849.ckpt
    model.load_state_dict(state_dict['state_dict'])

    # di = iter(dataloader)
    # datas,targets=next(di)
    saveDir=args.comment
    print(saveDir)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    fea_total=np.empty((0,768),np.float32)#256+512
    tar_total=np.empty(0,np.int64)
    if True:
        for i,(datas,targets) in enumerate(tqdm(dataloader)):
            result=model(datas)
            preds = result["logits_unlab"]
            preds_inc = torch.cat(
                [
                    result["logits_lab"].unsqueeze(1).expand(-1,args.num_heads, -1, -1),
                    result["logits_unlab"],
                ],
                dim=-1,
            )
            #preds = preds.max(dim=-1)[1]
            preds_inc = preds_inc.max(dim=-1)[1]
            preds_inc=preds_inc.permute((2,0,1))
            preds_inc=torch.reshape(preds_inc,(preds_inc.shape[0],-1))
            #collections.Counter(preds_inc)
            pp=[Counter(p).most_common(1)[0][0] for p in preds_inc.numpy()]#统计出现次数最多的标签
            pp=np.array(pp)
            f=result['feats'].max(0)[0]
            pfu = result['proj_feats_unlab'].max(0)[0].max(0)[0]
            cat_fea = torch.cat([f, pfu], dim=1).detach().numpy()
            #
            t=targets.numpy()
            fea_total=np.concatenate([fea_total,cat_fea],axis=0)
            tar_total=np.concatenate([tar_total,t],axis=0)

            print(float(np.equal(pp, t).sum()) / args.batch_size)

        np.savez(os.path.join(saveDir,'proj_numpy_ship'), x=fea_total, y=tar_total)
        # t_sne_projection(fea_total, tar_total)

        #keys: 'feats', 'logits_lab', 'logits_unlab', 'proj_feats_unlab', 'logits_unlab_over', 'proj_feats_unlab_over'

    cat_fea_total=np.empty((0,768),np.float32)
    cat_fea_target_total=np.empty(0,np.int64)
    for in_dataloader,valdataloader in enumerate(valdataloaders):
        cat_fea_sub = np.empty((0, 768), np.float32)
        cat_fea_target_sub = np.empty(0, np.int64)
        for i,(val_datas,val_targets) in enumerate(tqdm(valdataloader)):
            result=model(val_datas)
            preds = result["logits_unlab"]
            preds_inc = torch.cat(
                [
                    result["logits_lab"].unsqueeze(0).expand( args.num_heads, -1, -1),
                    result["logits_unlab"],
                ],
                dim=-1,
            )
            preds = preds.max(dim=-1)[1]
            preds_inc = preds_inc.max(dim=-1)[1]
            preds_inc = preds_inc.permute((1,0))
            #preds_inc = torch.reshape(preds_inc, (preds_inc.shape[0], -1))

            pp = [Counter(p).most_common(1)[0][0] for p in preds_inc.numpy()]
            pp = np.array(pp)
            f=result['feats']#.detach().numpy()
            pfu,_= result['proj_feats_unlab'].max(0)
            # pfu_n = pfu.detach().numpy()
            cat_fea=torch.cat([f,pfu],dim=1).detach().numpy()
            t = val_targets.numpy()
            cat_fea_sub=np.concatenate([cat_fea_sub, cat_fea],axis=0)
            cat_fea_target_sub = np.concatenate([cat_fea_target_sub, t], axis=0)
            print(float(np.equal(pp,t).sum())/args.batch_size)

        np.savez(os.path.join(saveDir,'proj_numpy_test_ship' + str(in_dataloader)), x=cat_fea_sub, y=cat_fea_target_sub)
        cat_fea_total = np.concatenate([cat_fea_total, cat_fea_sub], axis=0)
        cat_fea_target_total = np.concatenate([cat_fea_target_total, cat_fea_target_sub], axis=0)

    np.savez(os.path.join(saveDir,'proj_numpy_test_ship'), x=cat_fea_total, y=cat_fea_target_total)
    # t_sne_projection(cat_fea_total, cat_fea_target_total)
    #trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    #trainer.fit(model, dm)


def main_pretrain(args):
    dm = get_datamodule(args, "pretrain")
    dm.setup()
    dataloader=dm.train_dataloader(False)
    valdataloaders=dm.val_dataloader()#[val_subset_unlab_train, val_subset_unlab_test, val_subset_lab_test]

    model = Pretrainer(**args.__dict__)

    #state_dict = torch.load(args.pretrained)#  epoch=29-step=5849.ckpt
    #model.load_state_dict(state_dict['state_dict'])

    # di = iter(dataloader)
    # datas,targets=next(di)
    saveDir=args.comment
    print(saveDir)
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    fea_total=np.empty((0,768),np.float32)#256+512
    tar_total=np.empty(0,np.int64)
    if True:
        for i,(datas,targets) in enumerate(tqdm(dataloader)):
            result=model(datas)
            preds = result["logits_unlab"]
            preds_inc = torch.cat(
                [
                    result["logits_lab"].unsqueeze(1).expand(-1,args.num_heads, -1, -1),
                    result["logits_unlab"],
                ],
                dim=-1,
            )
            #preds = preds.max(dim=-1)[1]
            preds_inc = preds_inc.max(dim=-1)[1]
            preds_inc=preds_inc.permute((2,0,1))
            preds_inc=torch.reshape(preds_inc,(preds_inc.shape[0],-1))
            #collections.Counter(preds_inc)
            pp=[Counter(p).most_common(1)[0][0] for p in preds_inc.numpy()]#统计出现次数最多的标签
            pp=np.array(pp)
            f=result['feats'].max(0)[0]
            pfu = result['proj_feats_unlab'].max(0)[0].max(0)[0]
            cat_fea = torch.cat([f, pfu], dim=1).detach().numpy()
            #
            t=targets.numpy()
            fea_total=np.concatenate([fea_total,cat_fea],axis=0)
            tar_total=np.concatenate([tar_total,t],axis=0)

            print(float(np.equal(pp, t).sum()) / args.batch_size)

        np.savez(os.path.join(saveDir,'proj_numpy_ship_pretrain'), x=fea_total, y=tar_total)
        # t_sne_projection(fea_total, tar_total)

        #keys: 'feats', 'logits_lab', 'logits_unlab', 'proj_feats_unlab', 'logits_unlab_over', 'proj_feats_unlab_over'

    cat_fea_total=np.empty((0,768),np.float32)
    cat_fea_target_total=np.empty(0,np.int64)
    for in_dataloader,valdataloader in enumerate(valdataloaders):
        cat_fea_sub = np.empty((0, 768), np.float32)
        cat_fea_target_sub = np.empty(0, np.int64)
        for i,(val_datas,val_targets) in enumerate(tqdm(valdataloader)):
            result=model(val_datas)
            preds = result["logits_unlab"]
            preds_inc = torch.cat(
                [
                    result["logits_lab"].unsqueeze(0).expand( args.num_heads, -1, -1),
                    result["logits_unlab"],
                ],
                dim=-1,
            )
            preds = preds.max(dim=-1)[1]
            preds_inc = preds_inc.max(dim=-1)[1]
            preds_inc = preds_inc.permute((1,0))
            #preds_inc = torch.reshape(preds_inc, (preds_inc.shape[0], -1))

            pp = [Counter(p).most_common(1)[0][0] for p in preds_inc.numpy()]
            pp = np.array(pp)
            f=result['feats']#.detach().numpy()
            pfu,_= result['proj_feats_unlab'].max(0)
            # pfu_n = pfu.detach().numpy()
            cat_fea=torch.cat([f,pfu],dim=1).detach().numpy()
            t = val_targets.numpy()
            cat_fea_sub=np.concatenate([cat_fea_sub, cat_fea],axis=0)
            cat_fea_target_sub = np.concatenate([cat_fea_target_sub, t], axis=0)
            print(float(np.equal(pp,t).sum())/args.batch_size)

        np.savez(os.path.join(saveDir,'proj_numpy_test_ship_pretrain' + str(in_dataloader)), x=cat_fea_sub, y=cat_fea_target_sub)
        cat_fea_total = np.concatenate([cat_fea_total, cat_fea_sub], axis=0)
        cat_fea_target_total = np.concatenate([cat_fea_target_total, cat_fea_target_sub], axis=0)

    np.savez(os.path.join(saveDir,'proj_numpy_test_ship_pretrain'), x=cat_fea_total, y=cat_fea_target_total)
    # t_sne_projection(cat_fea_total, cat_fea_target_total)
    #trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    #trainer.fit(model, dm)
if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.max_epochs=1
    #main_discover(args)
    main_pretrain(args)

#--dataset CIFAR10 --gpus 1 --precision 16 --max_epochs 30 --batch_size 256 --num_labeled_classes 5 --num_unlabeled_classes 5 --pretrained checkpoints/epoch=29-step=5849.ckpt --num_heads 4 --comment 5_5

