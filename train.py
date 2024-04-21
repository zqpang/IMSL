from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random

import numpy as np
import numpy
import sys

import time
from datetime import timedelta
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F        

from imsl import datasets
from imsl import models
from imsl.models.memory_imsl import HybridMemory
from imsl.trainers import IMSL_USL
from imsl.evaluators import Evaluator, extract_cluster_features
from imsl.utils.data import IterLoader
from imsl.utils.data import transforms as T
from imsl.utils.data.sampler import RandomMultipleGallerySampler
from imsl.utils.data.preprocessor import Preprocessor, Train_Preprocessor
from imsl.utils.logging import Logger
from imsl.utils.serialization import load_checkpoint, save_checkpoint
from imsl.utils import mmc

from ChannelAug import ChannelExchange
from collections import OrderedDict
from itertools import chain

import os


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None, modal=1):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])
    
    train_transformer2 = T.Compose([
             T.Resize((height, width), interpolation=3),
             #T.Grayscale(num_output_channels=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
             ChannelExchange(gray = 2)
        ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances, modal)
    else:
        sampler = None
    
    if modal==2 or modal==3:
        train_transformer2 = None
    
    train_loader = IterLoader(
                DataLoader(Train_Preprocessor(train_set, root=None, transform1=train_transformer,transform2 = train_transformer2),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_query_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    train = True
    if (testset is None):
        testset = dataset.query
        train = False

    test_loader = DataLoader(
        Preprocessor(testset, train, root=None, transform1=test_transformer,transform2 = None),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader


def get_gallery_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    train = True
    if (testset is None):
        testset = dataset.gallery
        train = False

    test_loader = DataLoader(
        Preprocessor(testset, train, root=None, transform1=test_transformer,transform2 = None),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader



def get_cluster_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    train = True
    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))
        train = False

    test_loader = DataLoader(
        Preprocessor(testset, train, root=None, transform1=test_transformer,transform2 = None),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    return test_loader



def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    model.cuda()
    model = nn.DataParallel(model)
    return model

def evaluate_mean(evaluator1, dataset, test_loaders):
    maxap = 0
    maxcmc = 0
    mAP_sum = 0
    cmc_sum = 0
    cmc_sum_10 = 0

    for i in range(len(dataset)):
        cmc_scores, mAP = evaluator1.evaluate(test_loaders[i], dataset[i].query, dataset[i].gallery, cmc_flag=False)
        maxap = max(mAP, maxap)
        maxcmc = max(cmc_scores[0], maxcmc)
        mAP_sum += mAP
        cmc_sum += cmc_scores[0]
        cmc_sum_10 += cmc_scores[9]

    mAP = (mAP_sum) / len(test_loaders)
    cmc_now = (cmc_sum) / len(test_loaders)
    cmc_now_10 = cmc_sum_10 / (len(test_loaders))

    return mAP, cmc_now, cmc_now_10

def cluster_finement_new(index2label, pseudo_labels, rerank_dist, pseudo_labels_tight):
    rerank_dist_tensor = torch.tensor(rerank_dist)
    N = pseudo_labels.size(0)
    
    label_sim_expand = pseudo_labels.expand(N, N)
    label_sim_tight_expand = pseudo_labels_tight.expand(N, N)
    
    label_sim = label_sim_expand.eq(label_sim_expand.t()).float()
    label_sim_tight = label_sim_tight_expand.eq(label_sim_tight_expand.t()).float()
    
    sim_distance = rerank_dist_tensor.clone() * label_sim
    dists_labels = label_sim.sum(-1)
    
    dists_label_add = dists_labels.clone()
    dists_label_add[dists_label_add > 1] -= 1
    
    sim_add_average = sim_distance.sum(-1) / torch.pow(dists_labels, 2)
    
    cluster_I_average = torch.zeros(torch.max(pseudo_labels).item() + 1)
    for sim_dists, label in zip(sim_add_average, pseudo_labels):
        cluster_I_average[label.item()] += sim_dists
    
    sim_tight = label_sim.eq(1 - label_sim_tight.clone()).float()
    dists_tight = sim_tight * rerank_dist_tensor.clone()
    
    dists_label_tight_add = (1 + sim_tight.sum(-1))
    dists_label_tight_add[dists_label_tight_add > 1] -= 1
    
    sim_add_average = dists_tight.sum(-1) / torch.pow(dists_label_tight_add, 2)
    
    cluster_tight_average = torch.zeros(torch.max(pseudo_labels_tight).item() + 1)
    for sim_dists, label in zip(sim_add_average, pseudo_labels_tight):
        cluster_tight_average[label.item()] += sim_dists
    
    cluster_final_average = torch.zeros(len(sim_add_average))
    for i, label_tight in enumerate(pseudo_labels_tight):
        cluster_final_average[i] = cluster_tight_average[label_tight.item()]
    
    return cluster_final_average, cluster_I_average


def main():    
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)
    




def main_worker(args):
    global start_epoch, best_mAP
    best_mAP =0
    best_rank1 = 0
    start_time = time.monotonic()
    cudnn.benchmark = True
    
    sys.stdout = Logger(osp.join(args.logs_dir, args.dataset, 'imsl.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    query_loader = get_query_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    gallery_loader = get_gallery_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    
    
    all_img_cams = torch.tensor([c for _, _, c in sorted(dataset.train)])
    temp_all_cams = all_img_cams.numpy()
    all_img_cams = all_img_cams.cuda()
    unique_cameras = torch.unique(all_img_cams)
   
    
    model_rgb = create_model(args)
    
    
    evaluator1 = Evaluator(model_rgb)
    
    
    memory_rgb = HybridMemory(model_rgb.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum, all_img_cams=all_img_cams).cuda()
    memory_mask = HybridMemory(model_rgb.module.num_features, len(dataset.train),
                            temp=args.temp, momentum=args.momentum).cuda()


    cluster_loader1 = get_cluster_loader(dataset, args.height, args.width,
                                    64, args.workers, testset=sorted(dataset.train_rgb))

    cluster_loader2 = get_cluster_loader(dataset, args.height, args.width,
                                    64, args.workers, testset=sorted(dataset.train_ni))
    
    cluster_loader3 = get_cluster_loader(dataset, args.height, args.width,
                                    64, args.workers, testset=sorted(dataset.train_ti))
    

        
    params = []
    print('prepare parameter')
    
    models = [model_rgb]
    for model in models:
        for key, value in model.named_parameters():
            if value.requires_grad:
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]

    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    # Trainer
    print('==> Start training')
    trainer = IMSL_USL(model_rgb, memory_rgb, memory_mask)
    #start_score = 0
    
    
    for epoch in range(args.epochs):

        modal_labels = []
        for i, (fname, _, cid) in enumerate(sorted(dataset.train)):
            modal_labels.append(cid)
        modal_labels = torch.tensor(modal_labels)        
        
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        features_rgb, features2_rgb,  _ = extract_cluster_features(model_rgb, cluster_loader1, 1, print_freq=50)
        features_ni, features2_ni,  _ = extract_cluster_features(model_rgb, cluster_loader2, 2, print_freq=50)
        features_ti, features2_ti,  _ = extract_cluster_features(model_rgb, cluster_loader3, 3, print_freq=50)
        features = OrderedDict(chain(features_rgb.items(), features_ni.items(), features_ti.items()))
        features2 = OrderedDict(chain(features2_rgb.items(), features2_ni.items(), features2_ti.items()))
        
        del features_rgb, features2_rgb, features_ni, features2_ni, features_ti, features2_ti
        
        features = torch.cat([features[f.split('/')[-1][:-3]].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        features2 = torch.cat([features2[f.split('/')[-1][:-3]].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        #memory_mask.features = F.normalize(features2, dim=1).cuda()
        

        pseudo_labels = mmc.kmeans_with_modal(features, modal_labels, 200)
        
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        
        
        # generate proxy labels (camera-aware sub-cluster label)
        proxy_labels = -1 * np.ones(pseudo_labels.shape, pseudo_labels.dtype)
        cnt = 0
        for i in range(0, int(pseudo_labels.max() + 1)):
            inds = np.where(pseudo_labels == i)[0]
            local_cams = temp_all_cams[inds]
            for cc in np.unique(local_cams):
                pc_inds = np.where(local_cams==cc)[0]
                proxy_labels[inds[pc_inds]] = cnt
                cnt += 1
        num_proxies = len(set(proxy_labels)) - (1 if -1 in proxy_labels else 0)        
        
        
        # generate new dataset
        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label, accum_y) in enumerate(zip(sorted(dataset.train), pseudo_labels, proxy_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label, cid, i, accum_y)) 


        # statistics of clusters and un-clustered instances
        outlier_num = len(np.where(pseudo_labels==-1)[0])
        print('==> Statistics for epoch {}: {} clusters, {} sub-clusters, {} un-clustered instances'.
              format(epoch, num_ids, num_proxies, outlier_num))


        # re-initialize memory (pseudo label, memory feature and others)
        pseudo_labels = torch.from_numpy(pseudo_labels).long()
        memory_rgb.all_pseudo_label = pseudo_labels.cuda()
        memory_mask.all_pseudo_label = pseudo_labels.cuda()
        proxy_labels = torch.from_numpy(proxy_labels).long()
        memory_rgb.all_proxy_label = proxy_labels.cuda()
        
        


        memory_rgb.proxy_label_dict = {}  # {pseudo_label1: [proxy3, proxy10],...}
        for c in range(0, int(memory_rgb.all_pseudo_label.max() + 1)):
            memory_rgb.proxy_label_dict[c] = torch.unique(memory_rgb.all_proxy_label[memory_rgb.all_pseudo_label == c])
            

        memory_rgb.proxy_cam_dict = {}  # for computing proxy enhance loss
        for cc in unique_cameras:
            proxy_inds = torch.unique(memory_rgb.all_proxy_label[(all_img_cams == cc) & (memory_rgb.all_proxy_label>=0)])
            memory_rgb.proxy_cam_dict[int(cc)] = proxy_inds        
        
        
        #calculate proxy centers of images
        proxy_centers = torch.zeros(num_proxies, features.size(1))
        for lbl in range(num_proxies):
            ind = torch.nonzero(proxy_labels == lbl).squeeze(-1)  # note here
            id_feat = features[ind].mean(0)
            proxy_centers[lbl,:] = id_feat
        proxy_centers = F.normalize(proxy_centers.detach(), dim=1).cuda()
        print('  initializing proxy memory feature with shape {}...'.format(proxy_centers.shape))
        memory_rgb.global_memory = proxy_centers.detach()
        
        
        
        #calculate cluster centers of masks
        cluster_centers = torch.zeros(num_ids, features2.size(1))
        for lbl in range(num_ids):
            ind = torch.nonzero(pseudo_labels == lbl).squeeze(-1)  # note here
            id_feat = features2[ind].mean(0)
            cluster_centers[lbl,:] = id_feat
        cluster_centers = F.normalize(cluster_centers.detach(), dim=1).cuda()
        print('  initializing cluster memory feature with shape {}...'.format(cluster_centers.shape))
        memory_mask.global_memory = cluster_centers.detach()    

        
        
        train_loader1 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset, modal=1)

        train_loader2 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset, modal=2)

        train_loader3 = get_train_loader(args, dataset, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset, modal=3)

        train_loader1.new_epoch()
        train_loader2.new_epoch()
        train_loader3.new_epoch()
        
        trainer.train(epoch, train_loader1, train_loader2, train_loader3, optimizer,print_freq=args.print_freq, train_iters=len(train_loader1))
        
        
        if epoch > 30:
            args.eval_step = 1 
        
        
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            cmc_socore1,mAP1 = evaluator1.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, cmc_flag=False)
            mAP, cmc_now, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[9]
            
            
            print('===============================================')
            print('the RGB model performance')
            print('model mAP: {:5.1%}'.format(mAP))
            print('model cmc: {:5.1%}'.format(cmc_now))
            print('model cmc_10: {:5.1%}'.format(cmc_now_10))
            print('===============================================')
            
            
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            best_rank1 = max(cmc_now, best_rank1)
            
            
            save_checkpoint({
                'state_dict': model_rgb.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best,fpath=osp.join(args.logs_dir, args.dataset, 'checkpoint.pth.tar'))
            print('\n * Finished epoch {:3d}  model cmc: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, cmc_now, best_rank1, ' *' if is_best else ''))
        lr_scheduler.step()

        
        
    print ('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, args.dataset, 'model_best.pth.tar'))
    model_rgb.load_state_dict(checkpoint['state_dict'])
    
    cmc_socore1,mAP1 = evaluator1.evaluate(query_loader, gallery_loader, dataset.query, dataset.gallery, cmc_flag=False)
    mAP, cmc_now, cmc_now_10  = mAP1, cmc_socore1[0], cmc_socore1[9]
    
    print('=================RGB===================')
    print('the RGB model performance')
    print('model mAP: {:5.1%}'.format(mAP))
    print('model cmc: {:5.1%}'.format(cmc_now))
    print('model cmc_10: {:5.1%}'.format(cmc_now_10))
    print('===============================================')
    
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CICL")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='ltcc',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.60,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--output_weight', type=float, default=1.0,
                        help="loss outputs for weight ")
    parser.add_argument('--ratio_cluster', type=float, default=1.0,
                        help="cluster hypter ratio ")
    parser.add_argument('--cr', action="store_true", default=False,
                        help="use cluster refinement in CACL")
    
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--loss-size', type=int, default=2)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--sic_weight', type=float, default=1,
                        help="loss outputs for sic ")
    # training configs
    parser.add_argument('--seed', type=int, default=1)#
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    #data_dir = "/home/zhiqi/dataset"
    data_dir = "/home/userroot/database2/zhiqi/dataset"
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=data_dir)
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument("--cuda", type=str, default="0,1,2,3", help="cuda")
    main()
