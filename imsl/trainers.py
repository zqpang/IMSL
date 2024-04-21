from __future__ import print_function, absolute_import
import time
import torch
from .utils.meters import AverageMeter


class IMSL_USL(object):
    def __init__(self, encoder, memory1, memory2):
        super(IMSL_USL, self).__init__()
        self.encoder = encoder
        self.memory1 = memory1
        self.memory2 = memory2


    def train(self, epoch, data_loader1, data_loader2, data_loader3, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        
        losses1 = AverageMeter()
        losses2 = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs1 = data_loader1.next()
            inputs2 = data_loader2.next()
            inputs3 = data_loader3.next()
            
            
            data_time.update(time.time() - end)

            inputs_rgb, inputs_rgb2, cid_rgb, indexes1 = self._parse_data(inputs1)
            
            inputs_ni, inputs_ni2, cid_ni, indexes2 = self._parse_data(inputs2)
            
            inputs_ti, inputs_ti2, cid_ti, indexes3 = self._parse_data(inputs3)
            

            bn_x1, full_conect1, bn_x2, full_conect2 = self._forward(inputs_rgb, inputs_rgb2, inputs_ni, inputs_ni2, inputs_ti, inputs_ti2)
            
            #label = torch.cat((label_1, label_2), -1)
            indexes = torch.cat((indexes1, indexes2, indexes3), -1)
            cids = torch.cat((cid_rgb, cid_ni, cid_ti), -1)
            
            loss1 = self.memory1(bn_x1, full_conect2.clone(), indexes, cids, epoch, back = 1)
            loss2 = self.memory2(bn_x2, full_conect1.clone(), indexes, cids, epoch, back = 2)
 
            loss = (loss1 + loss2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses1.update(loss1.item())
            losses2.update(loss2.item())


            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.5f} ({:.5f})\t'
                      'Data {:.5f} ({:.5f})\t'
                      'Loss_ALL {:.5f} ({:.5f})\t'
                      'Loss_Image {:.5f} ({:.5f})\t'
                      'Loss_Mask {:.5f} ({:.5f})\t'
                      .format(epoch, i + 1, len(data_loader1),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses1.val, losses1.avg,
                              losses2.val, losses2.avg,

                              ))
                
                
    def _parse_data(self, inputs):
        img, img_mask, fname, pid, camid, index, accum_label = inputs
        return img.cuda(), img_mask.cuda(), camid.cuda(), index.cuda()

    def _forward(self, inputs_rgb, inputs_rgb2, inputs_ni, inputs_ni2, inputs_ti, inputs_ti2):
        
        bn_x1, full_conect1 = self.encoder(inputs_rgb, inputs_ni, inputs_ti, inputs_rgb, modal=0) #bn_x1.shape: torch.Size([64, 2048])
        
        bn_x2_rgb, full_conect2_rgb = self.encoder(inputs_rgb2, inputs_rgb2, inputs_rgb2, inputs_rgb2, modal=4)
        bn_x2_ir, full_conect2_ir = self.encoder(inputs_ni2, inputs_ni2, inputs_ni2, inputs_ni2, modal=4)
        bn_x2_ti, full_conect2_ti = self.encoder(inputs_ti2, inputs_ti2, inputs_ti2, inputs_ti2, modal=4)
        bn_x2 = torch.cat((bn_x2_rgb, bn_x2_ir, bn_x2_ti), 0)
        full_conect2 = torch.cat((full_conect2_rgb, full_conect2_ir, full_conect2_ti), 0)
        
        return bn_x1, full_conect1, bn_x2, full_conect2