# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x
class PCB(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice,cfg,part_output_size=(1,6)):
        super(PCB, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.use_global = cfg.MODEL.PCB_WITH_GLOBAL=='yes'
        if self.use_global:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.num_classes = num_classes
            self.neck = neck
            self.neck_feat = neck_feat

            if self.neck == 'no':
                self.classifier = nn.Linear(self.in_planes, self.num_classes)
    
            elif self.neck == 'bnneck':
                self.bottleneck = nn.BatchNorm1d(self.in_planes)
                self.bottleneck.bias.requires_grad_(False)  # no shift
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

                self.bottleneck.apply(weights_init_kaiming)
                self.classifier.apply(weights_init_classifier)
        # pcb
        self.pcb_adapool = nn.AdaptiveAvgPool2d(part_output_size)
        self.pcb_dropout = nn.Dropout(p=0.5)

        # define 6 classifiers
        
        self.part_output_size= part_output_size
        for i in range(part_output_size[0]*part_output_size[1]):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, num_classes, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        output = self.base(x)
        
        if self.training:
            ## pcb
            x = self.pcb_adapool(output)
            x = self.pcb_dropout(x)

            pcb_partial_predict = []
            # get six part feature batchsize*c
            for i in range(self.part_output_size[0]*self.part_output_size[1]):
                name = 'classifier'+str(i)
                c = getattr(self,name)
                if self.part_output_size[1]==1:
                    pcb_partial_predict.append(c(torch.squeeze(x[:,:,i])))
                else:
                    pcb_partial_predict.append(c(torch.squeeze(x[:,:,0,i])))
            if self.use_global:
                # global
                global_feat = self.gap(output)  # (b, 2048, 1, 1)
                global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

                if self.neck == 'no':
                    feat = global_feat
                elif self.neck == 'bnneck':
                    feat = self.bottleneck(global_feat)  # normalize for angular softmax
                # global
                cls_score = self.classifier(feat)
                pcb_partial_predict.append(cls_score)
                return torch.stack(pcb_partial_predict,dim=2), global_feat  # global feature for triplet loss
            return torch.stack(pcb_partial_predict,dim=2)  # global feature for triplet loss
        else:
            # pcb
            x = self.pcb_adapool(output)
            pcb_partial_predict = x.view(x.size(0),x.size(1),self.part_output_size[0]*self.part_output_size[1]) #G
            if self.use_global:
                # global
                   # global
                global_feat = self.gap(output)  # (b, 2048, 1, 1)
                global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

                if self.neck == 'no':
                    feat = global_feat
                elif self.neck == 'bnneck':
                    feat = self.bottleneck(global_feat)  # normalize for angular softmax
                if self.neck_feat == 'after':
                    # print("Test with feature after BN")
                    return torch.cat([pcb_partial_predict,feat.unsqueeze(-1)],dim=-1)
                else:
                    # print("Test with feature before BN")
                    return torch.cat([pcb_partial_predict,global_feat.unsqueeze(-1)],dim=-1)
            return pcb_partial_predict

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
