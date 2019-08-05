# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth,MarginSampleMiningLoss
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_softmargin':
        triplet = TripletLoss()
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_msml':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        msml = MarginSampleMiningLoss(cfg.SOLVER.MSML_MARGIN)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax' or cfg.MODEL.PCB_WITH_GLOBAL!='yes':
        print("using softmax loss")
        def loss_func(score, feat, target):
            if len(score.size()) == 3:
                target = target.unsqueeze(1).repeat(1,score.size()[-1])
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target)
            else:
                return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if len(score.size()) == 3:
                cls_target = target.unsqueeze(1).repeat(1,score.size()[-1])
            else:
                cls_target = target
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet' or cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_softmargin':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, cls_target) + triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, cls_target) + triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, cls_target) + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, cls_target) + cluster(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, cls_target) + triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0] + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, cls_target) + triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0] + cluster(feat, target)[0]    # new add by luo, no label smooth
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_msml':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, cls_target) + triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0] +cfg.SOLVER.MSML_LOSS_WEIGHT*msml(feat, target)[0]  # new add by yao, using msml
                else:
                    return F.cross_entropy(score, cls_target) + triplet(feat, target,batch_hard=cfg.MODEL.BATCH_HARD)[0] +cfg.SOLVER.MSML_LOSS_WEIGHT*msml(feat, target)[0]    # new add by yao, using msml
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048
    if 'PCB' in cfg.FRAMEWORK:
        assert cfg.MODEL.PCB_WITH_GLOBAL=='yes'

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_softmargin_center':
        triplet = TripletLoss()  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)
    
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_msml_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        msml = MarginSampleMiningLoss(cfg.SOLVER.MSML_MARGIN)

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, '
              'range_center,triplet_center, triplet_range_center '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        # import pdb
        # pdb.set_trace()
        if len(score.size()) == 3:
            cls_target = target.unsqueeze(1).repeat(1,score.size()[-1])
        else:
            cls_target = target
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, cls_target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, cls_target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)    # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, cls_target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0] # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, cls_target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]     # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center' or cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_softmargin_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, cls_target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, cls_target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)    # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, cls_target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, cls_target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]  # new add by luo, no label smooth
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_msml_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, cls_target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.MSML_LOSS_WEIGHT * msml(feat, target)[0]  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, cls_target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.MSML_LOSS_WEIGHT * msml(feat, target)[0]  # new add by luo, no label smooth
        else:
            print('expected METRIC_LOSS_TYPE with center should be center,'
                  ' range_center, triplet_center, triplet_range_center '
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion