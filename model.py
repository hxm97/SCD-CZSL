import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from backbone import Backbone
from basic_layers import MLP, DyCls

device = 'cuda:0'

class Model(nn.Module):
    def __init__(self, dset, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        
        self.num_atts = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx
        self.attr2idx = dset.attr2idx
        self.obj2idx = dset.obj2idx

        # Set training pairs.
        train_atts, train_objs = zip(*dset.train_pairs)
        train_atts = [dset.attr2idx[attr] for attr in train_atts]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        self.train_atts = torch.LongTensor(train_atts).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()
        
        self.feat_extractor = Backbone(cfg, 'resnet18')
        feat_dim = 512
        
        att_emb_modules = [
            nn.Conv2d(feat_dim, cfg.img_emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.img_emb_dim),
            nn.ReLU()
        ]
        obj_emb_modules = [
            nn.Conv2d(feat_dim, cfg.img_emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.img_emb_dim),
            nn.ReLU()
        ]
        pair_emb_modules = [
            nn.Conv2d(feat_dim, cfg.img_emb_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.img_emb_dim),
            nn.ReLU()
        ]
        
        if cfg.img_emb_drop > 0:
            att_emb_modules += [nn.Dropout2d(cfg.img_emb_drop)]
            obj_emb_modules += [nn.Dropout2d(cfg.img_emb_drop)]
            pair_emb_modules += [nn.Dropout2d(cfg.img_emb_drop)]
        
        self.att_embedder = nn.Sequential(*att_emb_modules)
        self.obj_embedder = nn.Sequential(*obj_emb_modules)
        self.pair_embedder = nn.Sequential(*pair_emb_modules)
        
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.att_mlp = MLP(cfg.img_emb_dim, cfg.img_emb_dim, cfg.img_emb_dim)
        self.obj_mlp = MLP(cfg.img_emb_dim, cfg.img_emb_dim, cfg.img_emb_dim)
        self.pair_mlp = MLP(cfg.img_emb_dim, cfg.img_emb_dim, cfg.img_emb_dim)
        self.att_cls = DyCls(cfg.img_emb_dim, self.num_atts, device, tmp = self.cfg.cosine_cls_temp)
        self.obj_cls = DyCls(cfg.img_emb_dim, self.num_objs, device, tmp = self.cfg.cosine_cls_temp)

        self.pair_cls = nn.Linear(cfg.img_emb_dim, len(dset.train_pairs))
        
        self.pair2att_cls = DyCls(cfg.img_emb_dim, self.num_atts, device, tmp = self.cfg.cosine_cls_temp)
        self.pair2obj_cls = DyCls(cfg.img_emb_dim, self.num_objs, device, tmp = self.cfg.cosine_cls_temp)
        
        self.enc_att = MLP(cfg.img_emb_dim, cfg.img_emb_dim, cfg.img_emb_dim)
        self.enc_obj = MLP(cfg.img_emb_dim, cfg.img_emb_dim, cfg.img_emb_dim)
        self.dec =  MLP(cfg.img_emb_dim*2, cfg.img_emb_dim, cfg.img_emb_dim)
        
            
    def train_forward(self, batch):
        
        img1 = batch['img']
        att_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']
        
        img1 = self.feat_extractor(img1)[0]
        bs = img1.shape[0]
        h, w = img1.shape[2:]
        
        img1_pair = self.pair_embedder(img1).view(bs, -1, h*w)
        img1_pair = self.img_avg_pool(img1_pair.view(bs, -1, h, w)).squeeze(-1).squeeze(-1)
        img1_pair = self.pair_mlp(img1_pair)
        img1_pair_pred = self.pair_cls(img1_pair)
        
        img1_att = self.att_embedder(img1).view(bs, -1, h*w)
        img1_att = self.img_avg_pool(img1_att.view(bs, -1, h, w)).squeeze(-1).squeeze(-1)
        img1_att = self.att_mlp(img1_att)
        att_pred = self.att_cls(img1_att, img1_pair)
        
        img1_obj = self.obj_embedder(img1).view(bs, -1, h*w)
        img1_obj = self.img_avg_pool(img1_obj.view(bs, -1, h, w)).squeeze(-1).squeeze(-1)
        img1_obj = self.obj_mlp(img1_obj)
        obj_pred = self.obj_cls(img1_obj, img1_pair)
        
        
        loss_att_base = F.cross_entropy(att_pred, att_labels)
        loss_obj_base = F.cross_entropy(obj_pred, obj_labels)
        loss_pair = F.cross_entropy(img1_pair_pred, pair_labels)
        
        img1_pair2att = self.enc_att(img1_pair)
        img1_pair2obj = self.enc_obj(img1_pair)
        img1_pair2att_pred = self.pair2att_cls(img1_pair2att, img1_pair)
        img1_pair2obj_pred = self.pair2obj_cls(img1_pair2obj, img1_pair)
    
        loss_att_pair = F.cross_entropy(img1_pair2att_pred, att_labels)
        loss_obj_pair = F.cross_entropy(img1_pair2obj_pred, obj_labels)
        loss_contra_att = contra(self.cfg.num_negs, img1_att, img1_pair2att, att_labels, temp=self.cfg.cosine_cls_temp)
        loss_contra_obj = contra(self.cfg.num_negs, img1_obj, img1_pair2obj, obj_labels, temp=self.cfg.cosine_cls_temp)
        
        img1_ao2pair = self.dec(torch.cat((img1_att, img1_obj), dim=1))
        img1_ao2pair_pred = self.pair_cls(img1_ao2pair)
        loss_ao_pair = F.cross_entropy(img1_ao2pair_pred, pair_labels) 
        
        out = {}
        out['loss_contra_att'] = self.cfg.w_loss_contra_att * loss_contra_att
        out['loss_contra_obj'] = self.cfg.w_loss_contra_obj * loss_contra_obj
        out['loss_att_pair'] = self.cfg.w_loss_att_pair * loss_att_pair
        out['loss_obj_pair'] = self.cfg.w_loss_obj_pair * loss_obj_pair
        out['loss_att_base'] = self.cfg.w_loss_att_base * loss_att_base
        out['loss_obj_base'] = self.cfg.w_loss_obj_base * loss_obj_base
        out['loss_pair'] = self.cfg.w_loss_pair * loss_pair
        out['loss_ao_pair'] = self.cfg.w_loss_ao_pair * loss_ao_pair
        out['loss_total'] = out['loss_contra_att']+out['loss_contra_obj']+out['loss_att_pair']+out['loss_obj_pair']+out['loss_att_base']+out['loss_obj_base'] +out['loss_pair']+out['loss_ao_pair']
        return out
    
    def val_forward(self, batch):
        img = batch['img']
        bs = img.shape[0]
        
        img = self.feat_extractor(img)[0]
        h, w = img.shape[2:]
        att = self.att_embedder(img).view(bs, -1, h*w)
        att = self.img_avg_pool(att.view(bs, -1, h, w)).squeeze(-1).squeeze(-1)
        att = self.att_mlp(att)
        
        obj = self.obj_embedder(img).view(bs, -1, h*w)
        obj = self.img_avg_pool(obj.view(bs, -1, h, w)).squeeze(-1).squeeze(-1)
        obj = self.obj_mlp(obj)
        
        pair = self.pair_embedder(img).view(bs, -1, h*w)
        pair = self.img_avg_pool(pair.view(bs, -1, h, w)).squeeze(-1).squeeze(-1)
        pair = self.pair_mlp(pair)
        att_pred = F.softmax(self.att_cls(att, pair), dim=1)
        obj_pred = F.softmax(self.obj_cls(obj, pair), dim=1)
        
        out = {}
        out['scores'] = {}
        for _, (att, obj) in enumerate(self.val_pairs):
            att_id, obj_id = self.attr2idx[att], self.obj2idx[obj]
            out['scores'][(att, obj)] = att_pred[:, att_id] * obj_pred[:, obj_id]
        return out
    
    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out

class CosineClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp
        self.concept = torch.Tensor(input_dim, output_dim).to(device)

    def forward(self, img, scale=True):
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(self.concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm)      
        if scale:
            pred = pred / self.temp
        return pred
        

def contra(num_negs, feat, neg, aim_labels, temp=0.05):
    
        feat, neg = F.normalize(feat, dim=-1), F.normalize(neg, dim=-1)
        batch_size, emb_dim = feat.shape[0], feat.shape[1]
        labels = torch.zeros(batch_size).long().to(device)
        pred = torch.zeros(batch_size, num_negs+1).to(device)
        
        for item in range(batch_size):
            neg_idx = sample_negative(item, num_negs, aim_labels)
            anchor = neg[neg_idx, :].squeeze(1)
            anchor = torch.cat((same_neg[item].unsqueeze(0), anchor), dim=0)
            anchor = torch.cat((neg[item, :].unsqueeze(0), anchor), dim=0)
            anchor = F.normalize(anchor, dim=1)
            pred[item, :] = torch.matmul(feat[item,:], anchor.transpose(0,1))
        pred = pred / temp
        loss = F.cross_entropy(pred, labels)
        return loss


def sample_negative(idx, num_negs, aim_labels):
    
    batch_size = aim_labels.shape[0]
    aim_ = aim_labels[idx]
    
    neg_num= 0
    pos_idx = np.arange(0, batch_size, 1)
    pos_idx = np.delete(pos_idx, idx, 0)
    neg_idx = []
    
    while neg_num < num_negs:
        
        neg_idx_ = np.random.choice(pos_idx, 1)
        neg_position = np.where(pos_idx==neg_idx_)
        while (aim_labels[neg_idx_] == aim_) :
            neg_idx_ = np.random.choice(pos_idx, 1)
            neg_position = np.where(pos_idx==neg_idx_)
        
        pos_idx = np.delete(pos_idx, neg_position, 0)
        neg_idx.append(neg_idx_)
        neg_num += 1
    
    return neg_idx

