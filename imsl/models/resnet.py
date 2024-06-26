from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

#from resnet_agw import resnet50 as resnet50_agw

import pdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']




class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.moda1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        self.moda2 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        self.moda3 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.moda4 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)        
        
        self.base = nn.Sequential(
            #resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)
        
        
        
        
        self.full_connect = nn.Linear(self.num_features, self.num_features)
        init.kaiming_normal_(self.full_connect.weight, mode='fan_out')
        init.constant_(self.full_connect.bias, 0)
        
        
        self.full_bn = nn.BatchNorm1d(self.num_features)
        self.full_bn.bias.requires_grad_(False)
        init.constant_(self.full_bn.weight, 1)
        init.constant_(self.full_bn.bias, 0)
        
        
        

        if not pretrained:
            self.reset_params()

    def forward(self, x1, x2, x3, x4, modal=0):
        
        if modal==0:
            x1 = self.moda1(x1)
            x2 = self.moda2(x2)
            x3 = self.moda3(x3)
            x = torch.cat((x1, x2, x3), 0)
        elif modal==1:
            x = self.moda1(x1)
        elif modal==2:
            x = self.moda2(x2)
        elif modal==3:
            x = self.moda3(x3)
        elif modal==4:
            #x = torch.cat((x1, x2, x3), 0)
            x = self.moda4(x4)
        
        #bs = x.size(0)
        x = self.base(x) #16 2048 16 8
        
        
        #gm_pool
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        p = 3.0
        x = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        
        #x = self.gap(x)
        #x = x.view(x.size(0), -1)


        if self.cut_at_pooling:
            return x

        
        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            dn_x = x
            bn_x = self.feat_bn(x)
            
            
        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x
        
        
        
        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)


        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            full_x = self.full_connect(dn_x)
            full_x = self.full_bn(full_x)
            full_x = F.normalize(full_x)

            return bn_x, full_x

        return prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
