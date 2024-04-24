import torch
import torch.nn.functional as F
from torch import nn, autograd

class ExemplarMemory(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None




class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, all_img_cams=''):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.all_img_cams = all_img_cams
        
        self.proxy_cam_dict = {}
        self.all_pseudo_label = ''
        self.all_proxy_label = ''
        self.proxy_label_dict = {}
        self.bg_knn = 40
        self.posK = 3
        self.balance_w = 0        

    def forward(self, features, mask_inputs_full, targets, cams, epoch, back=1):
        
        if back == 1:
            self.unique_cameras = torch.unique(self.all_img_cams)
            pseudo_y = self.all_pseudo_label[targets].to(torch.device('cuda'))
            proxy_targets = self.all_proxy_label[targets].to(torch.device('cuda'))
            loss = torch.tensor(0.).to(torch.device('cuda'))
            
            score = ExemplarMemory.apply(features, proxy_targets, self.global_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
            inputs = score / 0.10
            
            offline_loss, online_loss = 0, 0
            for cc in torch.unique(cams):
                inds = torch.nonzero(cams == cc).squeeze(-1)
                percam_inputs = inputs[inds]
                percam_y = pseudo_y[inds]
                pc_prx_target = proxy_targets[inds]
                
                offline_loss += self.get_proxy_associate_loss(percam_inputs, percam_y)
                
                temp_score = score[inds].detach().clone()  # for similarity
                online_loss += self.get_proxy_cam_wise_nn_enhance_loss(temp_score, percam_inputs, pc_prx_target)
                
            loss += (offline_loss + online_loss)
            
            return loss
            
        
        elif back == 2:
            # inputs: B*2048, features: L*2048
            targets1 = self.all_pseudo_label[targets].clone()
            
            old_inputs = features.clone()
            
            num_ids = self.all_pseudo_label.max()+1
            
            score = ExemplarMemory.apply(features, targets1, self.global_memory, torch.tensor(self.momentum).to(torch.device('cuda')))
            sim = score / self.temp
            
            soft_loss = self.soft_loss(sim.clone(), targets1.detach().clone(), num_ids)
            
            
            contras_loss = self.contrasloss(old_inputs.detach().clone(), mask_inputs_full.clone())
            
            return soft_loss + contras_loss*0.25
            


                    

    def focal_loss(self,targets ,sim, mask):
        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)
        
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
        targets_onehot = torch.zeros(masked_sim.size()).cuda()
        targets_squeeze = torch.unsqueeze(targets, 1)
        targets_onehot.scatter_(1, targets_squeeze, float(1))
        
        target_ones_p = targets_onehot.clone()
        focal_p  =target_ones_p.clone() * masked_sim.clone()
        focal_p_all = torch.pow(target_ones_p - focal_p, 2)

                
        outputs = torch.log(masked_sim+1e-6).float()
        loss = - (focal_p_all * outputs).float()
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)     
        return loss
    
    
    def soft_loss(self, inputs, targets, num_ids):
        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        log_probs = logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = 0.9 * targets + 0.1 / num_ids
        loss = (- targets * log_probs).mean(0).sum()
        return loss



    
    def contrasloss(self, inputs, another_inputs):
        inputs = (inputs.t() / inputs.norm(dim =1)).t()
        another_inputs = (another_inputs.t() / another_inputs.norm(dim =1)).t()
        loss = -1*(inputs * another_inputs).sum(dim = 1).mean()
        return loss


    def get_proxy_associate_loss(self, inputs, targets):
        temp_inputs = inputs.detach().clone()
        loss = 0
        for i in range(len(inputs)):
            pos_ind = self.proxy_label_dict[int(targets[i])]
            temp_inputs[i, pos_ind] = 10000.0  # mask the positives
            sel_ind = torch.sort(temp_inputs[i])[1][-self.bg_knn-len(pos_ind):]
            sel_input = inputs[i, sel_ind]
            sel_target = torch.zeros((len(sel_input)), dtype=sel_input.dtype).to(torch.device('cuda'))
            sel_target[-len(pos_ind):] = 1.0 / len(pos_ind)
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(inputs)
        return loss


    def get_proxy_cam_wise_nn_enhance_loss(self, temp_score, inputs, proxy_targets):

        temp_memory = self.global_memory.detach().clone()  # global_memory is the proxy memory
        soft_target = torch.zeros(self.bg_knn + self.posK, dtype=torch.float).to(torch.device('cuda'))
        soft_target[-self.posK:] = 1.0 / self.posK
        loss = 0

        for i in range(len(inputs)):
            lbl = proxy_targets[i]
            sims = self.balance_w * temp_score[i] + (1 - self.balance_w) * torch.matmul(temp_memory[lbl], temp_memory.t())

            all_cam_tops = []
            for cc in self.unique_cameras:
                proxy_inds = self.proxy_cam_dict[int(cc)]
                maxInd = sims[proxy_inds].argmax()  # obtain per-camera max
                all_cam_tops.append(proxy_inds[maxInd])

            # find the top-K inds among the per-camera max
            all_cam_tops = torch.tensor(all_cam_tops)
            sel_ind = torch.argsort(sims[all_cam_tops])[-self.posK:]
            sims[all_cam_tops[sel_ind]] = 10000  # mask positive proxies
            top_inds = torch.sort(sims)[1][-self.bg_knn-self.posK:]
            sel_input = inputs[i, top_inds]
            loss += -1.0 * (F.log_softmax(sel_input.unsqueeze(0), dim=1) * soft_target.unsqueeze(0)).sum()

        loss /= len(inputs)
        return loss
