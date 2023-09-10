import argparse
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from bisect import bisect_right

import evaluator_ge
import utils
from flags import parser, DATA_FOLDER
from dataset import CompositionDataset
from model import Model

best_auc, best_hm, best_seen, best_unseen = 0, 0, 0, 0

def freeze(m):
    """Freezes module m.
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None


def decay_learning_rate(optimizer, cfg):
    print('# of param groups in optimizer: %d' % len(optimizer.param_groups))
    param_groups = optimizer.param_groups
    for i, p in enumerate(param_groups):
        current_lr = p['lr']
        new_lr = current_lr * cfg.decay_factor
        print('Group %d: current lr = %.8f, decay to lr = %.8f' %(i, current_lr, new_lr))
        p['lr'] = new_lr


def decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg):
    milestones = cfg.lr_decay_milestones
    it = bisect_right(milestones, epoch)
    gamma = cfg.decay_factor ** it
    
    gammas = [gamma] * len(group_lrs)
    assert len(optimizer.param_groups) == len(group_lrs)
    i = 0
    for param_group, lr, gamma_group in zip(optimizer.param_groups, group_lrs, gammas):
        param_group["lr"] = lr * gamma_group
        i += 1
        print("Group %i, lr = %.8f" %(i, lr * gamma_group))


def train(epoch, model, optimizer, trainloader, device, cfg):
    model.train()
    freeze(model.feat_extractor)
    acc_attr_meter = utils.AverageMeter()
    acc_obj_meter = utils.AverageMeter()
    acc_pair_meter = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    end_time = time.time()
    
    start_iter = (epoch - 1) * len(trainloader)
    for idx, batch in enumerate(trainloader):
        it = start_iter + idx + 1
        data_time.update(time.time() - end_time)

        for k in batch:
            if isinstance(batch[k], list): 
                continue
            batch[k] = batch[k].to(device, non_blocking=True)
        
        out = model(batch)
        loss = out['loss_total']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate_ge(epoch, model, testloader, evaluator, device, topk=1):
    model.eval()

    dset = testloader.dataset
    val_attrs, val_objs = zip(*dset.pairs)
    val_attrs = [dset.attr2idx[attr] for attr in val_attrs]
    val_objs = [dset.obj2idx[obj] for obj in val_objs]
    model.val_attrs = torch.LongTensor(val_attrs).cuda()
    model.val_objs = torch.LongTensor(val_objs).cuda()
    model.val_pairs = dset.pairs

    all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], []
    for _, data in enumerate(testloader):
        for k in data:
            data[k] = data[k].to(device, non_blocking=True)

        out = model(data)
        predictions = out['scores']

        attr_truth, obj_truth, pair_truth = data['attr'], data['obj'], data['pair']

        all_pred.append(predictions)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = torch.cat(all_attr_gt).to('cpu'), torch.cat(all_obj_gt).to('cpu'), torch.cat(all_pair_gt).to('cpu')

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])

    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=topk)

    stats['a_epoch'] = epoch

    result = ''
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print('Test Epoch: %d' %epoch)
    print(result)

    del model.val_attrs
    del model.val_objs

    return stats['AUC'], stats['best_hm'], stats['best_seen'], stats['best_unseen']


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main_worker(gpu, cfg):
    
    print('Use GPU %d for training' %gpu)
    torch.cuda.set_device(gpu)
    device = 'cuda:%d'%gpu

    ckpt_dir = cfg.checkpoint_dir
    print(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    print('Batch size on each gpu: %d' %cfg.batch_size)
    print('Prepare dataset')
    trainset = CompositionDataset(
        phase='train', split=cfg.splitname, cfg=cfg)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker)
    
    valset = CompositionDataset(
        phase='val', split=cfg.splitname, cfg=cfg)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.test_batch_size, shuffle=False,
        num_workers=cfg.num_workers)
    
    testset = CompositionDataset(
        phase='test', split=cfg.splitname, cfg=cfg)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.test_batch_size, shuffle=False,
        num_workers=cfg.num_workers)

    model = Model(trainset, cfg)
    model.to(device)

    freeze(model.feat_extractor)

    evaluator_val_ge = evaluator_ge.Evaluator(valset, model, cfg)
    evaluator_test_ge = evaluator_ge.Evaluator(testset, model, cfg)
    
    torch.backends.cudnn.benchmark = True

    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        else:
            params.append(p)
            print('params_main: %s' % name)
        
    optimizer = optim.Adam([{'params': params, 'lr': cfg.lr},], lr=cfg.lr, weight_decay=cfg.wd)
    group_lrs = [cfg.lr]

    start_epoch = cfg.start_epoch
    epoch = start_epoch
    global best_auc, best_hm, best_seen, best_unseen
    
    while epoch <= cfg.max_epoch:
        train(epoch, model, optimizer, trainloader, device, cfg)

        if cfg.decay_strategy == 'milestone':
            decay_learning_rate_milestones(group_lrs, optimizer, epoch, cfg)

        if epoch < cfg.start_epoch_validate:
            epoch += 1
            continue

        if epoch % cfg.eval_every_epoch == 0:
            auc, hm, seen, unseen, attr, obj = validate_ge(epoch, model, testloader, evaluator_test_ge, device, topk=cfg.topk)

            if (auc > best_auc or auc / best_auc >= 0.99) and epoch == cfg.max_epoch and epoch+1 < cfg.final_max_epoch:
                cfg.max_epoch += 1

            if auc > best_auc:
                best_auc = auc
                print('New Best AUC', best_auc)
            
            if hm > best_hm:
                best_hm = hm
                print('New Best HM ', best_hm)
            
            if seen > best_seen:
                best_seen = seen
                print('New Best Seen ', best_seen)
            
            if unseen > best_unseen:
                best_unseen = unseen
                print('New Best Unseen ', best_unseen)
            
        epoch += 1
    
    print('Done: %s' % cfg.config_name)
    print('Best AUC %.4f HM %.4f Seen %.4f Unseen %.4f' %(best_auc, best_hm, best_seen, best_unseen))
                

def main():
    cfg = parser.parse_args()
    utils.load_args(cfg.config, cfg)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    main_worker(0, cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Best AUC %.4f HM %.4f Seen %.4f Unseen %.4f' %(best_auc, best_hm, best_seen, best_unseen))