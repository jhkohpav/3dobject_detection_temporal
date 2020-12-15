import time
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import spconv
import torchplus
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args
from second.pytorch.models.rgb_block import *
from torch.autograd import Variable
from numpy import meshgrid as n_mesh

class SECOND_FUSION_RPNV2_TEST(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 num_anchor_per_loc=2,
                 num_upsample_filters=[256, 256, 256],
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn',
                 use_grid_fusion=False,
                 fusion_grid_size=[6, 6, 6]):
        super(SECOND_FUSION_RPNV2_TEST, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        self._use_grid_fusion = use_grid_fusion
        self._fusion_grid_size = fusion_grid_size
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        n_feats = 128

        self.reg_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
        )
        self.cls_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=6, stride=1, padding=0),
            nn.ReLU(),
        )

        self.rgb_refine = Sequential(
           nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(n_feats),
           nn.ReLU(),
           nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
           nn.BatchNorm2d(n_feats),
           nn.ReLU(),
        )
        self.fusion_refine = Sequential(
           nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(n_feats*2),
           nn.ReLU(),
           nn.Conv2d(n_feats*2, n_feats, kernel_size=1, stride=1, padding=0),
           nn.BatchNorm2d(n_feats),
           nn.ReLU(),
        )

        self.bev_gate = BasicGate(n_feats)
        self.crop_gate = BasicGate(n_feats)

        self.conv_box_second = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, num_anchor_per_loc * box_code_size, stride=1, 
            padding=0, kernel_size=1),
        )
        self.conv_cls_second = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_feats, num_class, stride=1, padding=0, kernel_size=1),
        )
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, box_code_size)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                box_code_size)

    def forward(self, bev_feature, rgb_feat=None, ori_rgb_feat=None, box_3d_coords=None):
        crop_feature = self.rgb_refine(rgb_feat)
        
        bev_gated_s1 = self.bev_gate(bev_feature, bev_feature)
        rgb_gated_s1 = self.crop_gate(bev_feature, crop_feature)
        fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)
        fused_feat = self.fusion_refine(fused_feature_add)

        bev_feat_reg = self.reg_conv(bev_feature)
        bev_feat_cls = self.cls_conv(fused_feat)
        
        box_preds = self.conv_box_second(bev_feat_reg)
        cls_preds = self.conv_cls_second(bev_feat_cls)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        return ret_dict
        

class SECOND_FUSION_RPNV2(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 num_anchor_per_loc=2,
                 num_upsample_filters=[256, 256, 256],
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn',
                 use_grid_fusion=False,
                 fusion_grid_size=[6, 6, 6]):
        super(SECOND_FUSION_RPNV2, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        self._use_grid_fusion = use_grid_fusion
        self._fusion_grid_size = fusion_grid_size
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        n_feats = 128
        self.reg_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.cls_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )

        self.rgb_refine = Sequential(
           nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(128),
           nn.ReLU(),
           nn.Conv2d(128, n_feats, kernel_size=1, stride=1, padding=0),
           nn.BatchNorm2d(n_feats),
           nn.ReLU(),
        )
        self.fusion_refine = Sequential(
           nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm2d(n_feats*2),
           nn.ReLU(),
           nn.Conv2d(n_feats*2, n_feats, kernel_size=1, stride=1, padding=0),
           nn.BatchNorm2d(n_feats),
           nn.ReLU(),
        )

        self.bev_gate = BasicGate(n_feats)
        self.crop_gate = BasicGate(n_feats)
        NUM_FEATURES = 2
        if self._use_grid_fusion:
            NUM_FEATURES = 3
        # self.concat_bev_gate = BasicGate(n_feats*2)
        # self.concat_rgb_gate = BasicGate(n_feats*2)
        in_feat = sum(num_upsample_filters)
        self.conv_cls_second = nn.Conv2d(in_feat, num_class, 14)
        self.conv_box_second = nn.Conv2d(in_feat, num_anchor_per_loc * box_code_size, 14)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, box_code_size)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                box_code_size)

    def forward(self, bev_feature, rgb_feat=None, ori_rgb_feat=None, box_3d_coords=None):


        crop_feature = self.rgb_refine(rgb_feat)
        
        # fused_feature_wo_gating = torch.cat((bev_feature, crop_feature),dim=1)

        # bev_gated_s1 = self.concat_bev_gate(fused_feature_wo_gating, bev_feature)
        # rgb_gated_s1 = self.concat_rgb_gate(fused_feature_wo_gating, crop_feature)
        # fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)


        bev_gated_s1 = self.bev_gate(bev_feature, bev_feature)
        rgb_gated_s1 = self.crop_gate(bev_feature, crop_feature)
        #  rgb_gated_s1 = crop_feature
        #  bev_gated_s1 = bev_feature
        fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)
        fused_feat = self.fusion_refine(fused_feature_add)
        # concat_feat = fused_feat

        bev_feat_reg = self.reg_conv(bev_feature)
        bev_feat_cls = self.cls_conv(fused_feat)

        box_preds = self.conv_box_second(bev_feat_reg)
        cls_preds = self.conv_cls_second(bev_feat_cls)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        return ret_dict

class SECOND_FUSION_RPNV2_ori(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 num_anchor_per_loc=2,
                 num_upsample_filters=[256, 256, 256],
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(SECOND_FUSION_RPNV2, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        n_feats = 128
        self.reg_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.cls_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )

        self.rgb_refine = Sequential(
            nn.Conv2d(256*3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.fusion_refine = Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(),
            nn.Conv2d(n_feats*2, n_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )

        # self.bev_gate = BasicGate(n_feats)
        # self.crop_gate = BasicGate(n_feats)
        self.concat_bev_gate = BasicGate(n_feats*2)
        self.concat_rgb_gate = BasicGate(n_feats*2)
        in_feat = sum(num_upsample_filters)
        self.conv_cls_second = nn.Conv2d(in_feat, num_class, 6)
        self.conv_box_second = nn.Conv2d(in_feat, num_anchor_per_loc * box_code_size, 6)
        # import pdb; pdb.set_trace()
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, box_code_size)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                box_code_size)

    def forward(self, bev_feature, rgb_feat=None):

        crop_feature = self.rgb_refine(rgb_feat)
        
        fused_feature_wo_gating = torch.cat((bev_feature, crop_feature),dim=1)

        bev_gated_s1 = self.concat_bev_gate(fused_feature_wo_gating, bev_feature)
        rgb_gated_s1 = self.concat_rgb_gate(fused_feature_wo_gating, crop_feature)
        fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)


        # bev_gated_s1 = self.bev_gate(bev_feature, bev_feature)
        # rgb_gated_s1 = self.crop_gate(bev_feature, crop_feature)
        # rgb_gated_s1 = crop_feature
        # bev_gated_s1 = bev_feature
        # fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)
        fused_feat = self.fusion_refine(fused_feature_add)
        concat_feat = fused_feat

        bev_feat_reg = self.reg_conv(concat_feat)
        bev_feat_cls = self.cls_conv(concat_feat)

        box_preds = self.conv_box_second(bev_feat_reg)
        cls_preds = self.conv_cls_second(bev_feat_cls)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        return ret_dict

class RPN_SECOND_FUSION(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPN_SECOND_FUSION, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        
        for i, layer_num in enumerate(layer_nums):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        #########################
        det_num = sum(num_upsample_filters)
        #########################
        self.conv_cls = nn.Conv2d(det_num, num_cls, 1)
        self.conv_box = nn.Conv2d(
            det_num, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                det_num, num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                det_num, num_anchor_per_loc * box_code_size, 1)
        ##########################################################
        self.f_in_planes_det = 64
        net_type = 'FPN18'
        if net_type == 'FPN50':
            num_blocks = [3,4,6,3]
            bb_block = Bottleneck
        elif net_type == 'FPN18':
            num_blocks = [2,2,2,2]
            bb_block = BasicBlock

        # For RGB Feature Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer_det(bb_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_det(bb_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_det(bb_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_det(bb_block, 512, num_blocks[3], stride=2)
        if net_type == 'FPN18':
            fpn_sizes = [
                    self.layer2[1].conv2.out_channels,
                    self.layer3[1].conv2.out_channels,
                    self.layer4[1].conv2.out_channels]
        else:
            fpn_sizes = [self.layer2[num_blocks[1]-1].conv3.out_channels, self.layer3[num_blocks[2]-1].conv3.out_channels, self.layer4[num_blocks[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

    def _make_layer_det(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.f_in_planes_det, planes, stride))
            self.f_in_planes_det = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, f_view, idxs_norm, bev=None):

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        bev_feature = x
        ###################### FPN-18 ##########################
        with torch.no_grad():
            f1 = self.maxpool(F.relu(self.bn1(self.conv1(f_view))))
            f2 = self.layer1(f1)
            f3 = self.layer2(f2)
            f4 = self.layer3(f3)
            f5 = self.layer4(f4)
            f_view_features = self.fpn([f3, f4, f5])
            fuse_features2 = F.relu(f_view_features[0]).permute(0,1,3,2)
            fuse_features = fuse_features2.permute(0,1,3,2)
            #######################################################

            ##################### Z concat #########################
            crop_feature_all = []
            for i in range(idxs_norm.size()[1]):
                crop_feature_0 = feature_crop(fuse_features, idxs_norm[:,i,:,:])
                crop_feature_1 = feature_crop_interp(fuse-features, idxs_norm[:, i, :, :])
                import pdb; pdb.set_trace()
                crop_feature_all.append(crop_feature_0)
            #######################################################

            crop_feature_all = torch.stack(crop_feature_all, dim=2)
            N, C, D, H, W = crop_feature_all.shape
            crop_feature_all = crop_feature_all.view(N,C*D,H,W)

        ########################################
        box_preds = self.conv_box(bev_feature)
        cls_preds = self.conv_cls(bev_feature)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        ret_dict['gated_concat_feat'] = crop_feature_all.permute(1,0,2,3)
        ret_dict['gated_bev_feat'] = bev_feature.permute(1,0,2,3)

        return ret_dict
        

class SECOND_RPNV2(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 num_anchor_per_loc=2,
                 num_upsample_filters=[256, 256, 256],
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(SECOND_RPNV2, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        n_feats = 128
        self.reg_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.cls_conv = Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        in_feat = sum(num_upsample_filters)
        self.conv_cls_second = nn.Conv2d(in_feat, num_class, 6)
        self.conv_box_second = nn.Conv2d(in_feat, num_anchor_per_loc * box_code_size, 6)
        # import pdb; pdb.set_trace()
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, box_code_size)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                box_code_size)

    def forward(self, bev_feat, concat_feat=None):

        if concat_feat is None:
            concat_feat = bev_feat

        bev_feat_reg = self.reg_conv(bev_feat)
        bev_feat_cls = self.cls_conv(concat_feat)

        box_preds = self.conv_box_second(bev_feat_reg)
        cls_preds = self.conv_cls_second(bev_feat_cls)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        return ret_dict


class RPN_FUSION(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPN_FUSION, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        
        for i, layer_num in enumerate(layer_nums):
            # in_f = 256 if i == 0 else in_filters[i]
            in_f = in_filters[i]
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_f, num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        #########################
        det_num = sum(num_upsample_filters)
        #########################
        self.conv_cls = nn.Conv2d(det_num, num_cls, 1)
        self.conv_box = nn.Conv2d(
            det_num, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                det_num, num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                det_num, num_anchor_per_loc * box_code_size, 1)
        ##########################################################
        self.f_in_planes_det = 64
        net_type = 'FPN18'
        if net_type == 'FPN50':
            num_blocks = [3,4,6,3]
            bb_block = Bottleneck
        elif net_type == 'FPN18':
            num_blocks = [2,2,2,2]
            bb_block = BasicBlock

        # For RGB Feature Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer_det(bb_block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_det(bb_block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_det(bb_block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_det(bb_block, 512, num_blocks[3], stride=2)
        if net_type == 'FPN18':
            fpn_sizes = [
                    self.layer2[1].conv2.out_channels,
                    self.layer3[1].conv2.out_channels,
                    self.layer4[1].conv2.out_channels]
        else:
            fpn_sizes = [self.layer2[num_blocks[1]-1].conv3.out_channels, self.layer3[num_blocks[2]-1].conv3.out_channels, self.layer4[num_blocks[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        ####################################################################
        # Fusion Layer
        num_z_feat = 3
        n_feats = 128
        self.depth_refine = Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        # self.rgb_refine = Sequential(
        #     nn.Conv2d(256*num_z_feat, 256, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, n_feats, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(n_feats),
        #     nn.ReLU(),
        # )
        self.fusion_refine = Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_feats*2),
            nn.ReLU(),
            nn.Conv2d(n_feats*2, n_feats, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(n_feats),
            nn.ReLU(),
        )
        self.bev_gate = BasicGate(n_feats)
        self.crop_gate = BasicGate(n_feats)
        # self.depth_gate = BasicGate(n_feats)

    def _make_layer_det(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.f_in_planes_det, planes, stride))
            self.f_in_planes_det = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, f_view, idxs_norm,idxs_norm_ori, bev=None):
        # import pdb; pdb.set_trace()
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        ###################### FPN-18 ##########################
        # with torch.no_grad():
        bev_feature = x
        f1 = self.maxpool(F.relu(self.bn1(self.conv1(f_view))))
        f2 = self.layer1(f1)
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        f_view_features = self.fpn([f3, f4, f5])
        fuse_features2 = F.relu(f_view_features[0]).permute(0,1,3,2)
        fuse_features = fuse_features2.permute(0,1,3,2)
        
        
        c, r = n_mesh(np.arange(156), np.arange(48))
        # c, r = n_mesh(np.arange(1248), np.arange(384))
        points = np.stack([c, r])
        points = points.reshape((2, -1))
        points = points.T
        # fuse_features2
        B,_,W,H = bev_feature.shape
        projected = torch.zeros(B, 256, 200, 176).cuda() 

        for b_idx in range(B):
            idxs = idxs_norm[b_idx]
            fuse_features_b = fuse_features2[b_idx]
            projected[b_idx, :, idxs[:,1].to(torch.int64), idxs[:,0].to(torch.int64)] = fuse_features_b[:, points[:, 0].astype(np.int64), points[:, 1].astype(np.int64)]

        # crop_feature_all = []
        # for i in range(idxs_norm_ori.size()[1]):
        #     crop_feature_0 = feature_crop(fuse_features, idxs_norm_ori[:,i,:,:])
        #     crop_feature_all.append(crop_feature_0)
        # crop_feature_all = torch.stack(crop_feature_all, dim=2)
        # N, C, D, H, W = crop_feature_all.shape
        # crop_feature_all = crop_feature_all.view(N,C*D,H,W)

        # crop_feature = self.rgb_refine(crop_feature_all)
        projected_r = self.depth_refine(projected)

        bev_gated_s1 = self.bev_gate(bev_feature, bev_feature)
        # rgb_gated_s1 = self.crop_gate(bev_feature, crop_feature)
        depth_gated_s1 = self.crop_gate(bev_feature, projected_r)


        layer_list = []
        layer_list.append(projected.cpu().detach().numpy())
        layer_list.append(bev_feature.cpu().detach().numpy())
        layer_list.append(fuse_features.cpu().detach().numpy())
        layer_list.append(depth_gated_s1.cpu().detach().numpy())
        # layer_list.append(crop_feature_all.cpu().detach().numpy())
        layer_name = ['depth', 'bev', 'fused_rgb','depth_gated']
        fol_name='vis_depth/feature_vis'
        vis_feature(f_view, layer_list, layer_name, fol_name)

        import pdb; pdb.set_trace()

        # fused_feature_add = torch.cat((bev_gated_s1, rgb_gated_s1),dim=1)
        fused_feature_add = torch.cat((bev_gated_s1, depth_gated_s1),dim=1)

        fused_feat = self.fusion_refine(fused_feature_add)
        concat_feat = fused_feat
        
        ########################################
        box_preds = self.conv_box(bev_feature)
        cls_preds = self.conv_cls(concat_feat)

        # import pdb; pdb.set_trace()
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(concat_feat)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(reg_fused)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        ret_dict['gated_concat_feat'] = concat_feat.permute(1,0,2,3)
        ret_dict['gated_bev_feat'] = bev_feature.permute(1,0,2,3)

        # ret_dict["offset"] = self.off_input
        # import pdb; pdb.set_trace()
        return ret_dict




class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x, bev=None):
        # t = time.time()
        # torch.cuda.synchronize()

        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)

        return ret_dict

class RPNV2(nn.Module):
    """Compare with RPN, RPNV2 support arbitrary number of stage.
    """
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='rpn'):
        super(RPNV2, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._use_rc_net = use_rc_net
        # assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        """
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        """
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        deblocks = []
        
        for i, layer_num in enumerate(layer_nums):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(
                    in_filters[i], num_filters[i], 3, stride=layer_strides[i]),
                BatchNorm2d(num_filters[i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(
                    Conv2d(num_filters[i], num_filters[i], 3, padding=1))
                block.add(BatchNorm2d(num_filters[i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    num_filters[i],
                    num_upsample_filters[i],
                    upsample_strides[i],
                    stride=upsample_strides[i]),
                BatchNorm2d(num_upsample_filters[i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x, bev=None):
        # t = time.time()
        # torch.cuda.synchronize()
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))
        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)
        ret_dict['gated_bev_feat'] = x.permute(1,0,2,3)

        return ret_dict


class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(2)

class SparseRPN(nn.Module):
    """Don't use this.
    """
    def __init__(self,
                 output_shape,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 use_rc_net=False,
                 name='sparse_rpn'):
        super(SparseRPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self.name = name
        if use_norm:
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            SpConv3d = change_default_args(bias=False)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=False)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            BatchNorm1d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            SpConv3d = change_default_args(bias=True)(spconv.SparseConv3d)
            SubMConv3d = change_default_args(bias=True)(spconv.SubMConv3d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        # [11, 400, 352]
        self.block1 = spconv.SparseSequential(
            SpConv3d(
                num_input_features, num_filters[0], 3, stride=[2, layer_strides[0], layer_strides[0]], padding=[0, 1, 1]),
            BatchNorm1d(num_filters[0]),
            nn.ReLU())
        # [5, 200, 176]
        for i in range(layer_nums[0]):
            self.block1.add(SubMConv3d(
                num_filters[0], num_filters[0], 3, padding=1, indice_key="subm0"))
            self.block1.add(BatchNorm1d(num_filters[0]))
            self.block1.add(nn.ReLU())

        self.deconv1 = spconv.SparseSequential(
            SpConv3d(
                num_filters[0], num_filters[0], (3, 1, 1), stride=(2, 1, 1)),
            BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            SpConv3d(
                num_filters[0], num_upsample_filters[0], (2, 1, 1), stride=1),
            BatchNorm1d(num_upsample_filters[0]),
            nn.ReLU(),
            spconv.ToDense(),
            Squeeze()
        )  # [1, 200, 176]

        # [5, 200, 176]
        self.block2 = spconv.SparseSequential(
            SpConv3d(
                num_filters[0], num_filters[1], 3, stride=[2, layer_strides[1], layer_strides[1]], padding=[0, 1, 1]),
            BatchNorm1d(num_filters[1]),
            nn.ReLU())

        for i in range(layer_nums[1]):
            self.block2.add(SubMConv3d(
                num_filters[1], num_filters[1], 3, padding=1, indice_key="subm1"))
            self.block2.add(BatchNorm1d(num_filters[1]))
            self.block2.add(nn.ReLU())
        # [2, 100, 88]
        self.deconv2 = spconv.SparseSequential(
            SpConv3d(
                num_filters[1], num_filters[1], (2, 1, 1), stride=1),
            BatchNorm1d(num_filters[1]),
            nn.ReLU(),
            spconv.ToDense(),
            Squeeze(),
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU()
        )  # [1, 200, 176]

        self.block3 = spconv.SparseSequential(
            SpConv3d(
                num_filters[1], num_filters[2], [2, 3, 3], stride=[1, layer_strides[2], layer_strides[2]], padding=[0, 1, 1]),
            BatchNorm1d(num_filters[2]),
            nn.ReLU())

        for i in range(layer_nums[2]):
            self.block3.add(SubMConv3d(
                num_filters[2], num_filters[2], 3, padding=1, indice_key="subm2"))
            self.block3.add(BatchNorm1d(num_filters[2]))
            self.block3.add(nn.ReLU())
        

        self.deconv3 = Sequential(
            spconv.ToDense(),
            Squeeze(),
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        ) # [1, 200, 176]
        self.post = Sequential(
            Conv2d(
                sum(num_upsample_filters),
                128,
                3,
                stride=1,padding=1),
            BatchNorm2d(128),
            nn.ReLU(),
            Conv2d(
                128,
                64,
                3,
                stride=1,padding=1),
            BatchNorm2d(64),
            nn.ReLU(),

        ) # [1, 200, 176]
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        '''self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)
        '''
        self.conv_cls = nn.Conv2d(64, num_cls, 1)
        self.conv_box = nn.Conv2d(
            64, num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                64, num_anchor_per_loc * 2, 1)


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        sx = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        b1 = self.block1(sx)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        # print(b1.sparity, b2.sparity, b3.sparity)
        up1 = self.deconv1(b1)
        up2 = self.deconv2(b2)
        up3 = self.deconv3(b3)
        x = torch.cat([up1, up2, up3], dim=1)
        x = self.post(x)
        # out = self.to_dense(out).squeeze(2)
        # print("debug1")
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds

        return ret_dict
