from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
from itertools import chain

def extract_cnn_feature(model, inputs, modal):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs,inputs,inputs,inputs,modal)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features2 = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, imgs2, fnames, pids, cid, _)  in enumerate(data_loader):
            
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, cid[0].item())
            outputs_2 = extract_cnn_feature(model,imgs2, 4)
            for fname, output, output_2, pid in zip(fnames, outputs, outputs_2, pids):
                features[fname.split('/')[-1][:-3]] = output
                features2[fname.split('/')[-1][:-3]] = output_2
                labels[fname.split('/')[-1][:-3]] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, features2, labels



def extract_query_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features2 = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, imgs2, fnames, pids, cid, _)  in enumerate(data_loader):
            
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, 1)
            outputs_2 = extract_cnn_feature(model,imgs2, 4)
            for fname, output, output_2, pid in zip(fnames, outputs, outputs_2, pids):
                features[fname.split('/')[-1][:-3]] = output
                features2[fname.split('/')[-1][:-3]] = output_2
                labels[fname.split('/')[-1][:-3]] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, features2, labels



def extract_gallery_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features2 = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, imgs2, fnames, pids, cid, _)  in enumerate(data_loader):
            
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, 2)
            outputs_2 = extract_cnn_feature(model,imgs2, 4)
            for fname, output, output_2, pid in zip(fnames, outputs, outputs_2, pids):
                features[fname.split('/')[-1][:-3]] = output
                features2[fname.split('/')[-1][:-3]] = output_2
                labels[fname.split('/')[-1][:-3]] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, features2, labels


def extract_cluster_features(model, data_loader, modal, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    features2 = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, imgs2, fnames, pids, cid, _)  in enumerate(data_loader):
            
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, modal)
            outputs_2 = extract_cnn_feature(model,imgs2, 4)
            for fname, output, output_2, pid in zip(fnames, outputs, outputs_2, pids):
                features[fname.split('/')[-1][:-3]] = output
                features2[fname.split('/')[-1][:-3]] = output_2
                labels[fname.split('/')[-1][:-3]] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, features2, labels



def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f.split('/')[-1][:-3]].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f.split('/')[-1][:-3]].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid,  _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}
    
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, cmc_flag=False, rerank=False):
        features_query, _ , _ = extract_query_features(self.model, query_loader)
        features_gallery, _ , _ = extract_gallery_features(self.model, gallery_loader)
        features = OrderedDict(chain(features_query.items(), features_gallery.items()))
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)
