import sys
import time
from enum import Enum
from functools import reduce

import numpy as np
import spconv
import torch
import torchplus
from core.box_torch_ops import center_to_corner_box3d
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                        WeightedSmoothL1LocalizationLoss,
                                        WeightedSoftmaxClassificationLoss)
from second.pytorch.models import middle, rpn, voxel_encoder
from second.builder import similarity_calculator_builder
from torch import nn
from torch.nn import functional as F
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args

from solvers import ortools_solve
from utils_tr.data_util import get_start_gt_anno

import sys
sys.path.append('./mmMOT')
from modules.fusion_net import *
from modules.gcn import affinity_module
from modules.new_end import *
from functools import partial

sys.path.append('/mnt/new_iou/second.pytorch/second/Rotated_ROIAlign')
# sys.path.append('/home/spalab/jhyoo/15_second/second_v1.5/second.pytorch/second/Rotated_ROIAlign')
# sys.path.append('/media/hdd1/project/second/second_v1.5/new_iou/basecode_new_15_second/second.pytorch/second/Rotated_ROIAlign')
from roi_align_rotate import ROIAlignRotated
import second.data.kitti_common_tracking_vid as kitti
from modules.appear_net import AppearanceNet
from modules.point_net import *  # noqa
import torchvision.transforms as transforms
from point_cloud.box_np_ops import points_in_rbbox, box_camera_to_lidar, remove_outside_points


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_input_features=-1,
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_num_input_features=-1,
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_voxel_classifier=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 use_rc_net=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 measure_time=False,
                 name='voxelnet',
                 use_iou_branch=False,
                 iou_dict=None,
                 iou_loss_weight=1.0,
                 iou_loss_ftor=None,
                 use_iou_param_partaa=False,
                 criterion_tr=None, 
                 det_type='3D'                
                 ):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._second_nms_iou_threshold = 0.7

        # if self.training:
        self._second_nms_pre_max_size = 9000
        self._second_nms_post_max_size = 512

        # if True:
        #     self._second_nms_pre_max_size = 1024
        #     self._second_nms_post_max_size = 100
        #     self._nms_pre_max_size = 1024
        #     self._nms_post_max_size = 500


        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()
        self._use_iou_branch = use_iou_branch
        self._use_iou_param_partaa = use_iou_param_partaa

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._iou_loss_ftor = None
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        self._iou_loss_weight = iou_loss_weight
        if self._use_iou_branch:
            self._iou_loss_ftor = iou_loss_ftor
            self.iou = iou.get_iou_class(iou_dict.module_class_name)(
                num_filters=list(iou_dict.num_filters),
                num_input_features=iou_dict.num_input_features)
        self.measure_time = measure_time
        vfe_class_dict = {
            "VoxelFeatureExtractor": voxel_encoder.VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": voxel_encoder.VoxelFeatureExtractorV2,
            "VoxelFeatureExtractorV3": voxel_encoder.VoxelFeatureExtractorV3,
            "SimpleVoxel": voxel_encoder.SimpleVoxel
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        self.voxel_feature_extractor = vfe_class(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance)
        if len(middle_num_filters_d2) == 0:
            if len(middle_num_filters_d1) == 0:
                num_rpn_input_filters = vfe_num_filters[-1]
            else:
                num_rpn_input_filters = middle_num_filters_d1[-1]
        else:
            num_rpn_input_filters = middle_num_filters_d2[-1]

        if use_sparse_rpn: # don't use this. just for fun.
            self.sparse_rpn = rpn.SparseRPN(
                output_shape,
                # num_input_features=vfe_num_filters[-1],
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2,
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=num_rpn_input_filters * 2,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_bev=use_bev,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)
        else:
            mid_class_dict = {
                "SparseMiddleExtractor": middle.SparseMiddleExtractor,
                "SpMiddleD4HD": middle.SpMiddleD4HD,
                "SpMiddleD8HD": middle.SpMiddleD8HD,
                "SpMiddleFHD": middle.SpMiddleFHD,
                "SpMiddleFHDV2": middle.SpMiddleFHDV2,
                "SpMiddleFHDLarge": middle.SpMiddleFHDLarge,
                "SpMiddleResNetFHD": middle.SpMiddleResNetFHD,
                "SpMiddleD4HDLite": middle.SpMiddleD4HDLite,
                "SpMiddleFHDLite": middle.SpMiddleFHDLite,
                "SpMiddle2K": middle.SpMiddle2K,
            }
            mid_class = mid_class_dict[middle_class_name]
            self.middle_feature_extractor = mid_class(
                output_shape,
                use_norm,
                num_input_features=middle_num_input_features,
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
            rpn_class_dict = {
                "RPN": rpn.RPN,
                "RPNV2": rpn.RPNV2,
                "RPN_FUSION": rpn.RPN_FUSION,
                "RPN_SECOND_FUSION": rpn.RPN_SECOND_FUSION,
                "SECOND_RPNV2": rpn.SECOND_RPNV2,
                "SECOND_FUSION_RPNV2": rpn.SECOND_FUSION_RPNV2,
                "SECOND_FUSION_RPNV2_TEST": rpn.SECOND_FUSION_RPNV2_TEST,
            }
            self.rpn_class_name = rpn_class_name
            rpn_class = rpn_class_dict[self.rpn_class_name]
            self.rpn = rpn_class(
                use_norm=True,
                num_class=num_class,
                layer_nums=rpn_layer_nums,
                layer_strides=rpn_layer_strides,
                num_filters=rpn_num_filters,
                upsample_strides=rpn_upsample_strides,
                num_upsample_filters=rpn_num_upsample_filters,
                num_input_features=rpn_num_input_features,
                num_anchor_per_loc=target_assigner.num_anchors_per_location,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_bev=use_bev,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)
            if 'FUSION' in rpn_class_name:
                rpn_class = rpn_class_dict["SECOND_FUSION_RPNV2"]
                # rpn_class = rpn_class_dict["SECOND_FUSION_RPNV2_TEST"]
            else:
                rpn_class = rpn_class_dict["SECOND_RPNV2"]
            self.second_rpn = rpn_class(
                use_norm=True,
                num_class=num_class,
                num_anchor_per_loc=1,
                num_upsample_filters=rpn_num_upsample_filters,
                encode_background_as_zeros=encode_background_as_zeros,
                use_direction_classifier=use_direction_classifier,
                use_bev=use_bev,
                use_groupnorm=use_groupnorm,
                num_groups=num_groups,
                box_code_size=target_assigner.box_coder.code_size)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self._time_dict = {}
        self._time_total_dict = {}
        self._time_count_dict = {}

        # self.model_tr = model_tr
        self.criterion_tr = criterion_tr
        self.det_type = det_type
        self.used_id = []
        self.last_id = 0
        self.frames_id = []
        self.frames_det = []
        self.track_feats = None
        # if isinstance(model_tr, list):
        #     self.test_mode = model_tr[0].test_mode
        # else:
        # self.test_mode = model_tr.test_mode
        self.test_mode = 2

        self.lidar_conv = nn.Conv2d(256, 512, 14, 14)
        self.p_lidar_conv = nn.Conv2d(256, 512, 14, 14)
        self.relu = nn.ReLU(inplace=True)

        score_fusion_arch = 'A'
        fusion = eval(f"fusion_module_{score_fusion_arch}")
        self.fusion_module = fusion(
            512, 512, out_channels=512)
        self.w_det = nn.Sequential(
            nn.Conv1d(512, 512, 1, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512 // 2, 1, 1),
            nn.BatchNorm1d(512 // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(512 // 2, 1, 1, 1),
        )
        self.softmax_mode = 'single'
        new_end = partial(
            eval("NewEndIndicator_%s" % 'v2'),
            kernel_size=5,
            reduction=4,
            mode='avg')
        self.w_link = affinity_module(
            512, new_end=new_end, affinity_op='multiply')
        self.neg_threshold = 0.2

        self.appearance = AppearanceNet(
                'vgg',
                512,
                skippool=True,
                fpn=False,
                dropblock=0)
        point_net = eval("PointNet_v1")
        self.point_net = point_net(
                3,
                out_channels=512,
                use_dropout=False)

        self.center_limit_range = torch.tensor([0.0, -40.0, -3.0, 70.4, 40.0, 0.0]).cuda()

        self.sigmoid = nn.Sigmoid()
        self.conv_gating_bev = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.conv_gating_concat = nn.Conv2d(256, 1, kernel_size=3, padding=1)


    def clear_mem(self):
        self.used_id = []
        self.last_id = 0
        self.frames_id = []
        self.frames_det = []
        self.track_feats = None
        return

    def eval_tr(self):
        if isinstance(self.model_tr, list):
            for i in range(len(self.model_tr)):
                self.model_tr[i].eval()
        else:
            self.model_tr.eval()
        self.clear_mem()
        return

    def train_tr(self):
        if isinstance(self.model_tr, list):
            for i in range(len(self.model_tr)):
                self.model_tr[i].train()
        else:
            self.model_tr.train()
        self.clear_mem()
        return

    def mem_assign_det_id(self, feats, assign_det, assign_link, assign_new,
                          assign_end, det_split, dets):
        det_ids = []
        v, idx = torch.max(assign_link[0][0], dim=0)
        for i in range(idx.size(0)):
            if v[i] == 1:
                track_id = idx[i].item()
                det_ids.append(track_id)
                self.track_feats[track_id] = feats[i:i + 1]
            else:
                new_id = self.last_id + 1
                det_ids.append(new_id)
                self.last_id += 1
                self.track_feats.append(feats[i:i + 1])

        for k, v in dets[0].items():
            dets[0][k] = v.squeeze(0) if k != 'frame_idx' else v
        dets[0]['id'] = torch.Tensor(det_ids).long()
        self.frames_id.append(det_ids)
        self.frames_det += dets
        assert len(self.track_feats) == (self.last_id + 1)

        return det_ids, dets

    def align_id(self, dets_ids, dets_out):
        frame_start = 0
        if len(self.used_id) == 0:
            # Start of a sequence
            self.used_id += dets_ids
            self.frames_id += dets_ids
            self.frames_det += dets_out
            max_id = 0
            for i in range(len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    continue
                max_id = np.maximum(np.max(dets_ids[i]), max_id)
            self.last_id = np.maximum(self.last_id, max_id)
            return dets_ids, dets_out, frame_start
        elif self.frames_det[-1]['frame_idx'] != dets_out[0]['frame_idx']:
            # in case the sequence is not continuous
            aligned_ids = []
            aligned_dets = []
            max_id = 0
            id_offset = self.last_id + 1
            for i in range(len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    aligned_ids.append([])
                    continue
                new_id = dets_ids[i] + id_offset
                max_id = np.maximum(np.max(new_id), max_id)
                aligned_ids.append(new_id)
                dets_out[i]['id'] += id_offset
            aligned_dets += dets_out
            self.last_id = np.maximum(self.last_id, max_id)
            self.frames_id += aligned_ids
            self.frames_det += aligned_dets
            return aligned_ids, aligned_dets, frame_start
        else:
            # the first frame of current dets
            # and the last frame of last dets is the same
            frame_start = 1
            aligned_ids = []
            aligned_dets = []
            max_id = 0
            id_pairs = {}
            """
            assert len(dets_ids[0])== len(self.frames_id[-1])
            """
            # Calculate Id pairs
            for i in range(len(dets_ids[0])):
                # Use minimum because because sometimes
                # they are not totally the same
                has_match = False
                for j in range(len(self.frames_id[-1])):
                    if ((self.det_type == '3D'
                         and torch.sum(dets_out[0]['location'][i] !=
                                       self.frames_det[-1]['location'][j]) == 0
                         and torch.sum(dets_out[0]['bbox'][i] !=
                                       self.frames_det[-1]['bbox'][j]) == 0)
                            or (self.det_type == '2D' and torch.sum(
                                dets_out[0]['bbox'][i] != self.frames_det[-1]
                                ['bbox'][j]) == 0)):  # noqa

                        id_pairs[dets_ids[0][i]] = self.frames_id[-1][j]
                        has_match = True
                        break
                if not has_match:
                    id_pairs[dets_ids[0][i]] = self.last_id + 1
                    self.last_id += 1
            if len([v for k, v in id_pairs.items()]) != len(
                    set([v for k, v in id_pairs.items()])):
                print("ID pairs has duplicates!!!")
                print(id_pairs)
                print(dets_ids)
                print(dets_out[0])
                print(self.frames_id[-1])
                print(self.frames_det[-1])

            for i in range(1, len(dets_ids)):
                if dets_out[i]['id'].size(0) == 0:
                    aligned_ids.append([])
                    continue
                new_id = dets_ids[i].copy()
                for j in range(len(dets_ids[i])):
                    if dets_ids[i][j] in id_pairs.keys():
                        new_id[j] = id_pairs[dets_ids[i][j]]
                    else:
                        new_id[j] = self.last_id + 1
                        id_pairs[dets_ids[i][j]] = new_id[j]
                        self.last_id += 1
                if len(new_id) != len(
                        set(new_id)):  # check whether there is duplicate
                    print('have duplicates!!!')
                    print(id_pairs)
                    print(new_id)
                    print(dets_ids)
                    print(dets_out)
                    print(self.frames_id[-1])
                    print(self.frames_det[-1])
                    import pdb
                    pdb.set_trace()

                max_id = np.maximum(np.max(new_id), max_id)
                self.last_id = np.maximum(self.last_id, max_id)
                aligned_ids.append(new_id)
                dets_out[i]['id'] = torch.Tensor(new_id).long()
            # TODO: This only support check for 2 frame case
            if dets_out[1]['id'].size(0) != 0:
                aligned_dets += dets_out[1:]
                self.frames_id += aligned_ids
                self.frames_det += aligned_dets
            
            return aligned_ids, aligned_dets, frame_start

    def assign_det_id(self, assign_det, assign_link, assign_new, assign_end,
                      det_split, dets):
        det_start_idx = 0
        det_ids = []
        already_used_id = []
        fake_ids = []
        dets_out = []

        for i in range(len(det_split)):
            frame_id = []
            det_curr_num = det_split[i].item()
            fake_id = []
            det_out = get_start_gt_anno()
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # check w_det
                if assign_det[curr_det_idx] != 1:
                    fake_id.append(-1)
                    continue
                else:
                    # det_out.append(dets[i][j])
                    if dets[i][0]['name'][j] == 'Car':
                        class_num = torch.tensor([0])
                    det_out['name'].append(class_num)
                    det_out['truncated'].append(torch.tensor([dets[i][0]['truncated'][j]]))
                    det_out['occluded'].append(torch.tensor([dets[i][0]['occluded'][j]]))
                    det_out['alpha'].append(torch.tensor([dets[i][0]['alpha'][j]]))
                    det_out['bbox'].append(torch.tensor([dets[i][0]['bbox'][j]]))
                    det_out['dimensions'].append(torch.tensor([dets[i][0]['dimensions'][j]]))
                    det_out['location'].append(torch.tensor([dets[i][0]['location'][j]]))
                    det_out['rotation_y'].append(torch.tensor([dets[i][0]['rotation_y'][j]]))

                # w_det=1, check whether a new det
                if i == 0:
                    if len(already_used_id) == 0:
                        frame_id.append(0)
                        fake_id.append(0)
                        already_used_id.append(0)
                        det_out['id'].append(torch.Tensor([0]).long())
                    else:
                        new_id = already_used_id[-1] + 1
                        frame_id.append(new_id)
                        fake_id.append(new_id)
                        already_used_id.append(new_id)
                        det_out['id'].append(torch.Tensor([new_id]).long())
                    continue
                elif assign_new[curr_det_idx] == 1:
                    new_id = already_used_id[-1] + 1 if len(
                        already_used_id) > 0 else 0
                    frame_id.append(new_id)
                    fake_id.append(new_id)
                    already_used_id.append(new_id)
                    det_out['id'].append(torch.Tensor([new_id]).long())
                else:
                    # look prev
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if assign_link[i - 1][0][k][j] == 1:
                            prev_id = fake_ids[-1][k]
                            frame_id.append(prev_id)
                            fake_id.append(prev_id)
                            det_out['id'].append(
                                torch.Tensor([prev_id]).long())
                            break

            assert len(fake_id) == det_curr_num
            fake_ids.append(fake_id)
            det_ids.append(np.array(frame_id))
            for k, v in det_out.items():
                if len(det_out[k]) == 0:
                    det_out[k] = torch.Tensor([])
                else:
                    det_out[k] = torch.cat(v, dim=0)

            det_out['frame_idx'] = ['%06d'%int(dets[i][0]['image_idx'][0][5:])]
            dets_out.append(det_out)
            det_start_idx += det_curr_num
        return det_ids, dets_out

    def start_timer(self, *names):
        if not self.measure_time:
            return
        for name in names:
            self._time_dict[name] = time.time()
        torch.cuda.synchronize()

    def end_timer(self, name):
        if not self.measure_time:
            return
        torch.cuda.synchronize()
        time_elapsed = time.time() - self._time_dict[name]
        if name not in self._time_count_dict:
            self._time_count_dict[name] = 1
            self._time_total_dict[name] = time_elapsed
        else:
            self._time_count_dict[name] += 1
            self._time_total_dict[name] += time_elapsed
        self._time_dict[name] = 0

    def clear_timer(self):
        self._time_count_dict.clear()
        self._time_dict.clear()
        self._time_total_dict.clear()

    def get_avg_time_dict(self):
        ret = {}
        for name, val in self._time_total_dict.items():
            count = self._time_count_dict[name]
            ret[name] = val / max(1, count)
        return ret

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def similarity_fn_wrapper(self, similarity_fn, boxes1, boxes2):
        # calculate bev iou
        boxes1_rbv = boxes1[:, [0, 1, 3, 4, 6]].cpu().detach().numpy()
        boxes2_rbv = boxes2[:, [0, 1, 3, 4, 6]].cpu().detach().numpy()
        boxes_iou = similarity_fn.compare(boxes1_rbv, boxes2_rbv)
        # calculate z overlap 
        boxes1_z = boxes1[:, [2, 5]].cpu().detach().numpy()
        boxes2_z = boxes2[:, [2, 5]].cpu().detach().numpy()
        boxes1_z[:, [0, 1]] = np.vstack([boxes1_z[:, 0] - (boxes1_z[:, 1]/2), boxes1_z[:, 0] + (boxes1_z[:, 1]/2)]).T
        boxes2_z[:, [0, 1]] = np.vstack([boxes2_z[:, 0] - (boxes2_z[:, 1]/2), boxes2_z[:, 0] + (boxes2_z[:, 1]/2)]).T
        mask_overlap = np.max(np.stack(np.array([boxes1_z[:, 1], boxes2_z[:, 1]])), axis=0) \
                        - np.min(np.stack(np.array([boxes1_z[:, 0], boxes2_z[:, 0]])), axis=0) \
                        - (boxes1_z[:, 1] - boxes1_z[:, 0]) \
                        - (boxes2_z[:, 1] - boxes2_z[:, 0]) >= 0
        ovr_z = np.stack(np.array([boxes1_z[:, 1] - boxes2_z[:, 0], boxes2_z[:, 1] - boxes1_z[:, 0]]).T).min(axis=1)
        ovr_z[mask_overlap] = 0.
        # calculate abs iou value
        a = torch.mul(boxes1[:, 3], boxes1[:, 4]).cpu().detach().numpy()
        b = torch.mul(boxes2[:, 3], boxes2[:, 4]).cpu().detach().numpy()
        d = boxes_iou.copy()
        ovr_bev = np.divide(np.multiply(a, d) + np.multiply(b, d), 1 + d)
        # return 3d iou
        ovr_3d = np.multiply(ovr_bev, ovr_z)
        a = np.multiply(a, boxes1[:, 5].cpu().detach().numpy())
        b = np.multiply(b, boxes2[:, 5].cpu().detach().numpy())
        iou_3d = np.divide(ovr_3d, a + b - ovr_3d)
        return iou_3d

    def forward(self, example, train_param):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        f_view = example["f_view"]
        idxs_norm = example["idxs_norm"]

        p_voxels = example["p_voxels"]
        p_coors = example["p_coordinates"]
        p_f_view = example["p_f_view"]
        p_idxs_norm = example["p_idxs_norm"]
        p_num_points = example["p_num_points"]

        batch_size_dev = batch_anchors.shape[0]
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        # t = time.time()

        self.start_timer("voxel_feature_extractor")
        voxel_features = self.voxel_feature_extractor(voxels, num_points)
        p_voxel_features = self.voxel_feature_extractor(p_voxels, p_num_points)
        self.end_timer("voxel_feature_extractor")
        # torch.cuda.synchronize()
        # print("vfe time", time.time() - t)

        # import pdb; pdb.set_trace()
        self.start_timer("middle forward")
        spatial_features = self.middle_feature_extractor(voxel_features, coors, batch_size_dev)
        p_spatial_features = self.middle_feature_extractor(p_voxel_features, p_coors, batch_size_dev)
        self.end_timer("middle forward")
        self.start_timer("rpn forward")

        # import pdb; pdb.set_trace()
        feats_bev = torch.cat([p_spatial_features, spatial_features], dim=1)
        channel = p_spatial_features.shape[1]
        gating_weight1 = self.sigmoid(self.conv_gating_bev(feats_bev))
        gating_weight2 = 1. - gating_weight1
        feats_bev_gated = feats_bev[:,:channel,:,:]*gating_weight1 + feats_bev[:,channel:channel*2,:,:]*gating_weight2

        # feats_concat = torch.cat([p_concat_crops_output, concat_crops_output], dim=1)
        # gating_weight1_concat = self.sigmoid(self.conv_gating_concat(feats_concat))
        # gating_weight2_concat = 1. - gating_weight1_concat
        # feats_concat_gated = feats_concat[:,:channel,:,:]*gating_weight1_concat + feats_concat[:,channel:channel*2,:,:]*gating_weight2_concat

        if 'FUSION' in self.rpn_class_name:
            preds_dict = self.rpn(feats_bev_gated, f_view, idxs_norm)
            # p_preds_dict = self.rpn(p_spatial_features, p_f_view, p_idxs_norm)
        else:
            preds_dict = self.rpn(spatial_features)

        self.end_timer("rpn forward")

        if self._use_iou_branch:
            iou_preds = self.iou(preds_dict['feature'])
            preds_dict['iou_preds'] = iou_preds
        else:
            iou_preds = None

        box_preds = preds_dict["box_preds"]
        cls_preds = preds_dict["cls_preds"]
        dir_preds = preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)
        # p_box_preds = p_preds_dict["box_preds"]
        # p_cls_preds = p_preds_dict["cls_preds"]
        # p_dir_preds = p_preds_dict["dir_cls_preds"].view(batch_size_dev, -1, 2)
        # import pdb; pdb.set_trace()
        if train_param:
            labels = example['labels']
            reg_targets = example['reg_targets']

            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            loc_loss, cls_loss, _ = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight
            loss = loc_loss_reduced + cls_loss_reduced

        if train_param:
            labels = example['labels']
            reg_targets = example['reg_targets']

        ################### ROI start #############################
        res, box_3d_coords, selected_idx, box_index, each_size = self.predict_for_rpn(example, preds_dict)

        boxes = center_to_corner_box3d(box_3d_coords[...,:3],box_3d_coords[...,3:6],angles=torch.zeros_like(box_3d_coords[..., 6]), axis=1, origin=[0.5, 0.5, 0])
        boxes[...,0] /= 70.4
        boxes[...,1] += 40.
        boxes[...,1] /= 80.
        ymin, xmin, ymax, xmax = boxes[...,0].min(dim=1)[0], boxes[...,1].min(dim=1)[0],  boxes[...,0].max(dim=1)[0], boxes[...,1].max(dim=1)[0]

        boxes = torch.stack((xmin,ymin,xmax,ymax),dim=-1)
        ymin *= 176.
        ymax *= 176.
        xmin *= 200.
        xmax *= 200.
        height = xmax-xmin
        width = ymax-ymin
        x_center = xmin + height/2.
        y_center = ymin + width/2.
        angles = -box_3d_coords[..., 6]*180/(2*torch.asin(torch.tensor(1.)))
        boxes = torch.stack((box_index.to(torch.float32).cuda(), y_center, x_center, width, height, angles),dim=-1)

        # p_res, p_box_3d_coords, p_selected_idx, p_box_index, p_each_size = self.predict_for_rpn(example, p_preds_dict)

        # p_boxes = center_to_corner_box3d(p_box_3d_coords[...,:3],p_box_3d_coords[...,3:6],angles=torch.zeros_like(p_box_3d_coords[..., 6]), axis=1, origin=[0.5, 0.5, 0])
        # p_boxes[...,0] /= 70.4
        # p_boxes[...,1] += 40.
        # p_boxes[...,1] /= 80.
        # p_ymin, p_xmin, p_ymax, p_xmax = p_boxes[...,0].min(dim=1)[0], p_boxes[...,1].min(dim=1)[0],  p_boxes[...,0].max(dim=1)[0], p_boxes[...,1].max(dim=1)[0]

        # p_boxes = torch.stack((p_xmin,p_ymin,p_xmax,p_ymax),dim=-1)
        # p_ymin *= 176.
        # p_ymax *= 176.
        # p_xmin *= 200.
        # p_xmax *= 200.
        # p_height = p_xmax-p_xmin
        # p_width = p_ymax-p_ymin
        # p_x_center = p_xmin + p_height/2.
        # p_y_center = p_ymin + p_width/2.
        # p_angles = -p_box_3d_coords[..., 6]*180/(2*torch.asin(torch.tensor(1.)))
        # p_boxes = torch.stack((p_box_index.to(torch.float32).cuda(), p_y_center, p_x_center, p_width, p_height, p_angles),dim=-1)

        bev_crop_input = preds_dict['gated_bev_feat'].permute(1,0,2,3).contiguous()
        pooler_rotated1 = ROIAlignRotated((14,14), spatial_scale = (1.), sampling_ratio = 0)
        bev_crops_output = pooler_rotated1(bev_crop_input,boxes)

        # p_bev_crop_input = p_preds_dict['gated_bev_feat'].permute(1,0,2,3).contiguous()
        # p_pooler_rotated1 = ROIAlignRotated((14,14), spatial_scale = (1.), sampling_ratio = 0)
        # p_bev_crops_output = p_pooler_rotated1(p_bev_crop_input, p_boxes)

        concat_crops_output = None
        if 'FUSION' in self.rpn_class_name:
            concat_crop_input = preds_dict['gated_concat_feat'].permute(1,0,2,3).contiguous()
            pooler_rotated2 = ROIAlignRotated((14,14), spatial_scale = (1.), sampling_ratio = 0)
            concat_crops_output = pooler_rotated2(concat_crop_input,boxes)

            # p_concat_crop_input = p_preds_dict['gated_concat_feat'].permute(1,0,2,3).contiguous()
            # p_pooler_rotated2 = ROIAlignRotated((14,14), spatial_scale = (1.), sampling_ratio = 0)
            # p_concat_crops_output = p_pooler_rotated2(p_concat_crop_input,p_boxes)

        # import pdb; pdb.set_trace()
        # feats_bev = torch.cat([p_bev_crops_output, bev_crops_output], dim=1)
        # channel = p_bev_crops_output.shape[1]
        # gating_weight1 = self.sigmoid(self.conv_gating_bev(feats_bev))
        # gating_weight2 = 1. - gating_weight1
        # feats_bev_gated = feats_bev[:,:channel,:,:]*gating_weight1 + feats_bev[:,channel:channel*2,:,:]*gating_weight2

        # feats_concat = torch.cat([p_concat_crops_output, concat_crops_output], dim=1)
        # gating_weight1_concat = self.sigmoid(self.conv_gating_concat(feats_concat))
        # gating_weight2_concat = 1. - gating_weight1_concat
        # feats_concat_gated = feats_concat[:,:channel,:,:]*gating_weight1_concat + feats_concat[:,channel:channel*2,:,:]*gating_weight2_concat

        second_preds_dict = self.second_rpn(bev_crops_output, concat_crops_output)
        # p_second_preds_dict = self.second_rpn(p_bev_crops_output, p_concat_crops_output)

        second_box_preds = second_preds_dict['box_preds']
        second_cls_preds = second_preds_dict['cls_preds']
        # p_second_box_preds = p_second_preds_dict['box_preds']
        # p_second_cls_preds = p_second_preds_dict['cls_preds']

        first_box_preds = box_preds.view(batch_size_dev,-1,self._box_coder.code_size)
        first_cls_preds = cls_preds.view(batch_size_dev,-1,1)
        selected_idx = selected_idx.to(torch.int64)
        batch_first_box_preds = []
        batch_first_cls_preds = []
        second_anchors = []
        for b_idx in range(batch_size_dev):
            sdx = sum(each_size[:b_idx])
            edx = sum(each_size[:b_idx+1])
            second_anchors.append(example['anchors'][b_idx][selected_idx[sdx:edx]])
            batch_first_box_preds.append(first_box_preds[b_idx][selected_idx[sdx:edx]])
            batch_first_cls_preds.append(first_cls_preds[b_idx][selected_idx[sdx:edx]])

        second_anchors = torch.stack(second_anchors,dim=0).view(batch_size_dev,each_size[0],self._box_coder.code_size)
        batch_first_cls_preds = torch.stack(batch_first_cls_preds,dim=0).view(batch_size_dev, each_size[0],1)
        batch_first_box_preds = torch.stack(batch_first_box_preds,dim=0).view(batch_size_dev, each_size[0],self._box_coder.code_size)
        second_box_preds = second_box_preds.view(batch_size_dev, each_size[0],self._box_coder.code_size)

        second_box_preds += batch_first_box_preds
        second_cls_preds = second_cls_preds.view(batch_size_dev, each_size[0],1)

        # p_first_box_preds = p_box_preds.view(batch_size_dev,-1,self._box_coder.code_size)
        # p_first_cls_preds = p_cls_preds.view(batch_size_dev,-1,1)
        # p_selected_idx = p_selected_idx.to(torch.int64)
        # p_batch_first_box_preds = []
        # p_batch_first_cls_preds = []
        # p_second_anchors = []
        # for b_idx in range(batch_size_dev):
        #     sdx = sum(each_size[:b_idx])
        #     edx = sum(each_size[:b_idx+1])
        #     p_second_anchors.append(example['anchors'][b_idx][p_selected_idx[sdx:edx]])
        #     p_batch_first_box_preds.append(p_first_box_preds[b_idx][p_selected_idx[sdx:edx]])
        #     p_batch_first_cls_preds.append(p_first_cls_preds[b_idx][p_selected_idx[sdx:edx]])

        # p_second_anchors = torch.stack(p_second_anchors,dim=0).view(batch_size_dev,each_size[0],self._box_coder.code_size)
        # p_batch_first_cls_preds = torch.stack(p_batch_first_cls_preds,dim=0).view(batch_size_dev, each_size[0],1)
        # p_batch_first_box_preds = torch.stack(p_batch_first_box_preds,dim=0).view(batch_size_dev, each_size[0],self._box_coder.code_size)
        # p_second_box_preds = p_second_box_preds.view(batch_size_dev, each_size[0],self._box_coder.code_size)

        # p_second_box_preds += p_batch_first_box_preds
        # p_second_cls_preds = p_second_cls_preds.view(batch_size_dev, each_size[0],1)
        preds_dict['box_preds'] = batch_first_box_preds
        preds_dict['cls_preds'] = batch_first_cls_preds
        # p_preds_dict['box_preds'] = p_batch_first_box_preds
        # p_preds_dict['cls_preds'] = p_batch_first_cls_preds
        c_box_preds = self.decoding_box_cur(example, preds_dict, second_anchors, selected_idx)
        c_box_preds_nms, c_selected = self.nms_vid(c_box_preds, preds_dict['cls_preds'])
        # p_box_preds_nms, p_selected = self.nms_vid(p_box_preds, p_preds_dict['cls_preds'])


        if train_param:
            _, _, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)
            selected_reg_targets = []
            selected_cls_targets = []
            selected_labels = []
            selected_idx = selected_idx.to(torch.int64)

            for b_idx in range(batch_size_dev):
                sdx = sum(each_size[:b_idx])
                edx = sum(each_size[:b_idx+1])
                selected_reg_targets.append(example['reg_targets'][b_idx][selected_idx[sdx:edx]])
                selected_cls_targets.append(cls_targets[b_idx][selected_idx[sdx:edx]])
                selected_labels.append(labels[b_idx][selected_idx[sdx:edx]])

            selected_reg_targets = torch.stack(selected_reg_targets,dim=0).view(batch_size_dev, each_size[0],self._box_coder.code_size)
            selected_cls_targets = torch.stack(selected_cls_targets,dim=0).view(batch_size_dev,each_size[0],1)
            selected_labels = torch.stack(selected_labels,dim=0).view(batch_size_dev, each_size[0])

            if self._use_iou_branch or self._use_iou_param_partaa:
                reg_targets_ = selected_reg_targets.clone()
                batch_anchors_ = second_anchors.clone()
                labels_ = selected_labels.clone()
                box_preds_ = second_box_preds.clone()
                box_preds_ = box_preds_.view(batch_size_dev, -1,
                                            self._box_coder.code_size)
                similarity_fn = similarity_calculator_builder.build_iouloss("rotate_iou_similarity")
                
                if self._use_iou_branch:
                    iou_preds = iou_preds.view(batch_size_dev, -1, 1)
                iou_targets = torch.zeros((batch_size_dev, box_preds_.shape[1], 1), dtype=torch.float32).cuda()
                for i in range(batch_size_dev):
                    # get positive anchors idx
                    pos_idx = (labels_[i]>0).nonzero().squeeze()
                    try:
                        if pos_idx.shape[0] == 0: continue
                    except:
                        pos_idx = torch.tensor([0]).cuda()
                    # compute preds, targs decode coordinates
                    box_preds_decode = self._box_coder.decode_torch(box_preds_[i][pos_idx], batch_anchors_[i][pos_idx])
                    reg_targets_decode = self._box_coder.decode_torch(reg_targets_[i][pos_idx], batch_anchors_[i][pos_idx])
                    # compute overlaps between the anchors and the targets
                    preds_by_targ_iou = self.similarity_fn_wrapper(similarity_fn, box_preds_decode, reg_targets_decode)
                    # allocate iou_targets
                    if self._use_iou_param_partaa:
                        # from Part-A^2 Equation 13.
                        iou_1 = preds_by_targ_iou > 0.75
                        iou_2 = preds_by_targ_iou < 0.25
                        iou_3 = np.logical_not(np.logical_or(iou_1, iou_2))
                        preds_by_targ_iou[iou_1] = 1.
                        preds_by_targ_iou[iou_2] = 0.
                        preds_by_targ_iou[iou_3] = preds_by_targ_iou[iou_3]*2 - 0.5
                    iou_targets[i][pos_idx, 0] = torch.tensor(preds_by_targ_iou).cuda()
            else:
                iou_preds, iou_targets = None, None

            cls_weights, reg_weights, cared = prepare_loss_weights(
                selected_labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            if self._use_iou_param_partaa:
                selected_cls_targets = selected_cls_targets.to(torch.float) * iou_targets
            loc_loss_second, cls_loss_second, iou_loss_second = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=second_box_preds,
                cls_preds=second_cls_preds,
                cls_targets=selected_cls_targets,
                cls_weights=cls_weights,
                reg_targets=selected_reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            second_batch = sum(each_size)
            loc_loss_reduced_second = loc_loss_second.sum() / batch_size_dev
            loc_loss_reduced_second *= self._loc_loss_weight
            cls_pos_loss_second, cls_neg_loss_second = _get_pos_neg_loss(cls_loss_second, selected_labels)
            cls_pos_loss_second /= self._pos_cls_weight
            cls_neg_loss_second /= self._neg_cls_weight
            cls_loss_reduced_second = cls_loss_second.sum() / batch_size_dev
            cls_loss_reduced_second *= self._cls_loss_weight
            if self._use_iou_branch:
                iou_loss_reduced = iou_loss_second.sum() / batch_size_dev
                iou_loss_reduced *= self._iou_loss_weight
            else:
                iou_loss_reduced = iou_loss_second
            loss_second = loc_loss_reduced_second + cls_loss_reduced_second + iou_loss_reduced
            if 'FUSION' in self.rpn_class_name:
                concat_out = concat_crops_output.permute(1,0,2,3)
            else:
                concat_out = None
            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'],
                                                reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].view(
                    batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(
                    dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight

                second_dir_targets = get_direction_target(second_anchors,
                                                selected_reg_targets)
                second_dir_logits = second_preds_dict["dir_cls_preds"].view(
                    batch_size_dev, -1, 2)
                weights = (selected_labels > 0).type_as(second_dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                second_dir_loss = self._dir_loss_ftor(
                    second_dir_logits, second_dir_targets, weights=weights)
                second_dir_loss = second_dir_loss.sum() / batch_size_dev
                loss_second += second_dir_loss * self._direction_loss_weight

                    # calculate loss
                res_sum = {
                    "labels" : selected_labels,
                    "loss": (loss_second+loss)/2.,
                    "cls_loss": cls_loss_second,
                    "loc_loss": loc_loss_second,
                    "cls_pos_loss": cls_pos_loss_second,
                    "cls_neg_loss": cls_neg_loss_second,
                    "cls_preds": second_cls_preds,
                    "dir_loss_reduced": dir_loss,
                    "cls_loss_reduced": (cls_loss_reduced_second+cls_loss_reduced)/2.,
                    "loc_loss_reduced": (loc_loss_reduced_second+loc_loss_reduced)/2.,
                    "cared": cared,
                    # "bev_crops_output": bev_crops_output.permute(1,0,2,3),
                    # "concat_crops_output": concat_out,
                }

            res = {
                "labels" : selected_labels,
                "loss": loss_second,
                "cls_loss": cls_loss_second,
                "loc_loss": loc_loss_second,
                "cls_pos_loss": cls_pos_loss_second,
                "cls_neg_loss": cls_neg_loss_second,
                "cls_preds": second_cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced_second,
                "loc_loss_reduced": loc_loss_reduced_second,
                "cared": cared,
                # "bev_crops_output": bev_crops_output.permute(1,0,2,3),
                # "concat_crops_output": concat_out,
            }

            if self._use_iou_branch:
                res["iou_loss"] = iou_loss
                res["iou_loss_reduced"] = iou_loss_reduced
                res_sum["iou_loss"] = iou_loss
                res_sum["iou_loss_reduced"] = iou_loss_reduced
            return res_sum
        else:
            preds_dict['box_preds'] = second_box_preds
            preds_dict['cls_preds'] = second_cls_preds 
            self.start_timer("predict")
            res = self.predict_v2(example, preds_dict, second_anchors, selected_idx)
            self.end_timer("predict")
            return res


    def predict_for_rpn(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_coder.code_size)
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds, batch_anchors)
        batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        bwise_box3d_lidar = []
        bwise_selected_idx = []
        each_size = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            rect = rect.float()
            Trv2c = Trv2c.float()
            P2 = P2.float()
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            # nms_func = box_torch_ops.rotate_nms
            nms_func = box_torch_ops.nms

            # get highest score per prediction, than apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)


            # if self._nms_score_threshold > 0.0:
            # thresh = torch.tensor(
            #     [0.3],
            #     device=total_scores.device).type_as(total_scores)
            # top_scores_keep = (top_scores >= thresh)
            # top_scores = top_scores.masked_select(top_scores_keep)


            if top_scores.shape[0] != 0:
                # if self._nms_score_threshold > 0.0:
                # box_preds = box_preds[top_scores_keep]

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                    boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                    box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._second_nms_pre_max_size,
                    post_max_size=self._second_nms_post_max_size,
                    iou_threshold=self._second_nms_iou_threshold,
                )
                selected_boxes = box_preds[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            else:
                selected = []
                selected_boxes = box_preds[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]

        # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels

                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                    "selected_idx" : selected
                }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "bbox": torch.zeros([0, 4], dtype=dtype, device=device),
                    "box3d_camera": torch.zeros([0, 7], dtype=dtype, device=device),
                    "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
            bwise_box3d_lidar.append(final_box_preds)
            bwise_selected_idx.append(selected)

        bwise_box3d_lidar = torch.cat(bwise_box3d_lidar,dim=0)

        bwise_selected_idx = torch.cat(bwise_selected_idx,dim=0)
        bwise_idx = torch.arange(0,batch_size)
        bwise_box_idx = []
        for b_idx in range(batch_size):
            b_idxs = bwise_idx[b_idx].repeat(predictions_dicts[b_idx]['bbox'].shape[0])
            bwise_box_idx.append(b_idxs)
            each_size.append(predictions_dicts[b_idx]['bbox'].shape[0])
        bwise_box_idx = torch.cat(bwise_box_idx)
        # each_size = torch.cat(each_size)
        return predictions_dicts, bwise_box3d_lidar, bwise_selected_idx, bwise_box_idx, each_size


    def predict_v2(self, example, preds_dict, batch_anchors, selected_idx):
        t = time.time()

        batch_size = batch_anchors.shape[0]
        batch_anchors = batch_anchors.view(batch_size, -1, self._box_coder.code_size)

        # batch_size = example['anchors'].shape[0]
        # batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
        batch_imgidx = example['image_idx']

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        if 'iou_preds' in preds_dict.keys():
            batch_iou_preds = preds_dict["iou_preds"]
        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        if self._use_iou_branch:
            batch_iou_preds = batch_iou_preds.view(batch_size, -1,
                                                   num_class_with_bg)
        else:
            batch_iou_preds = torch.zeros_like(batch_cls_preds)
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        dir_idx = 0
        for box_preds, cls_preds, iou_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_iou_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask):
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
                iou_preds = iou_preds[a_mask]
            box_preds = box_preds.float()
            cls_preds = cls_preds.float()
            iou_preds = iou_preds.float()
            rect = rect.float()
            Trv2c = Trv2c.float()
            P2 = P2.float()
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                # print(dir_preds.shape)
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
                total_ious = torch.sigmoid(iou_preds)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                selected_boxes = torch.cat(selected_boxes, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_scores = torch.cat(selected_scores, dim=0)
                if self._use_direction_classifier:
                    selected_dir_labels = torch.cat(
                        selected_dir_labels, dim=0)
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_ious = total_ious.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)

                #  if self._nms_score_threshold > 0.0:
                    #  thresh = torch.tensor(
                        #  [self._nms_score_threshold],
                        #  device=total_scores.device).type_as(total_scores)
                    #  top_scores_keep = (top_scores >= thresh)
                    #  top_scores = top_scores.masked_select(top_scores_keep)

                if self._nms_score_threshold > 0.0:
                    if self._use_iou_branch:
                        top_ious_keep = top_ious >= self._nms_score_threshold
                        top_ious = top_ious.masked_select(top_ious_keep)
                        top_scores = top_scores.masked_select(top_ious_keep)
                        top_scores_keep = top_ious_keep
                    else:
                        top_scores_keep = top_scores >= self._nms_score_threshold
                        top_scores = top_scores.masked_select(top_scores_keep)
                        top_ious = top_scores.clone()

                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[selected_idx[512*dir_idx:512*(dir_idx+1)]][top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_ious,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )

                else:
                    selected = []

                # if selected is not None:
                selected_boxes = box_preds[selected]
                if self._use_direction_classifier:
                    selected_dir_labels = dir_labels[selected]
                selected_labels = top_labels[selected]
                selected_scores = top_scores[selected]
            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "bbox": torch.zeros([0, 4], dtype=dtype, device=device),
                    "box3d_camera": torch.zeros([0, 7], dtype=dtype, device=device),
                    "box3d_lidar": torch.zeros([0, 7], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0, 4], dtype=top_labels.dtype, device=device),
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
            dir_idx +=1
        return predictions_dicts

    def decoding_box(self, example, preds_dict, batch_anchors, selected_idx, p_preds_dict, p_batch_anchors, p_selected_idx):
        t = time.time()

        batch_size = batch_anchors.shape[0]
        batch_anchors = batch_anchors.view(batch_size, -1, self._box_coder.code_size)
        p_batch_anchors = p_batch_anchors.view(batch_size, -1, self._box_coder.code_size)

        # batch_size = example['anchors'].shape[0]
        # batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        batch_imgidx = example['image_idx']
        p_batch_imgidx = example['p_image_idx']

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)

        p_batch_box_preds = p_preds_dict["box_preds"]
        p_batch_cls_preds = p_preds_dict["cls_preds"]
        p_batch_box_preds = p_batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        p_batch_box_preds = self._box_coder.decode_torch(p_batch_box_preds,
                                                       p_batch_anchors)

        return batch_box_preds, p_batch_box_preds

    def decoding_box_cur(self, example, preds_dict, batch_anchors, selected_idx):
        t = time.time()

        batch_size = batch_anchors.shape[0]
        batch_anchors = batch_anchors.view(batch_size, -1, self._box_coder.code_size)
        # p_batch_anchors = p_batch_anchors.view(batch_size, -1, self._box_coder.code_size)

        # batch_size = example['anchors'].shape[0]
        # batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)

        batch_imgidx = example['image_idx']
        # p_batch_imgidx = example['p_image_idx']

        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)
        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)

        # p_batch_box_preds = p_preds_dict["box_preds"]
        # p_batch_cls_preds = p_preds_dict["cls_preds"]
        # p_batch_box_preds = p_batch_box_preds.view(batch_size, -1,
        #                                        self._box_coder.code_size)
        # p_batch_box_preds = self._box_coder.decode_torch(p_batch_box_preds,
        #                                                p_batch_anchors)

        return batch_box_preds

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled,
                    #    loss_tr,
                       vox_preds=None,
                       vox_labels=None,
                       vox_weights=None):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
            # "loss_tr": float(loss_tr.cpu().detach().numpy()),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    def predict(self, det_imgs, det_info, dets, det_split):
        feats = []
        appear = self.appearance(det_imgs)
        feats.append(appear)
        feats = torch.cat(feats, dim=-1).t().unsqueeze(0)
        points, trans = self.point_net(
                det_info['points'].transpose(-1, -2),
                det_info['points_split'].long().squeeze(0))

        feats = torch.cat([feats, points.transpose(1,0).unsqueeze(0)], dim=1)
        tracking_fusion_feats = self.fusion_module(feats)
        det_scores = self.determine_det(tracking_fusion_feats, False)

        start = 0
        link_scores = []
        new_scores = []
        end_scores = []
        for i in range(len(det_split) - 1):
            prev_end = start + det_split[i].item()
            end = prev_end + det_split[i + 1].item()
            link_score, new_score, end_score = self.associate(
                tracking_fusion_feats[:, :, start:prev_end], tracking_fusion_feats[:, :, prev_end:end])
            link_scores.append(link_score.squeeze(1))
            new_scores.append(new_score)
            end_scores.append(end_score)
            start = prev_end

        fake_new = det_scores.new_zeros(
            (det_scores.size(0), link_scores[0].size(-2)))
        fake_end = det_scores.new_zeros(
            (det_scores.size(0), link_scores[-1].size(-1)))
        new_scores = torch.cat([fake_new] + new_scores, dim=1)
        end_scores = torch.cat(end_scores + [fake_end], dim=1)

        # det_score, link_score, new_score, end_score, _ = self.model_tr(
            # det_imgs, det_info, det_split)

        assign_det, assign_link, assign_new, assign_end = ortools_solve(
            det_scores[self.test_mode],
            [link_scores[0][self.test_mode:self.test_mode + 1]],
            new_scores[self.test_mode], end_scores[self.test_mode], det_split)

        assign_id, assign_bbox = self.assign_det_id(assign_det, assign_link,
                                                    assign_new, assign_end,
                                                    det_split, dets)
        aligned_ids, aligned_dets, frame_start = self.align_id(
            assign_id, assign_bbox)

        return aligned_ids, aligned_dets, frame_start

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(child)
        return net

    def determine_det(self, feats, train_param):
        det_scores = self.w_det(feats).squeeze(1)  # Bx1xL -> BxL

        if not train_param:
            # add mask
            if 'cls' in 'branch_cls':
                det_scores = det_scores.sigmoid()
#             print(det_scores[:, -1].size())
#             mask = det_scores[:, -1].lt(self.neg_threshold)
#             det_scores[:, -1] -= mask.float()
            mask = det_scores.lt(self.neg_threshold)
            det_scores -= mask.float()
        return det_scores

    def associate(self, objs, dets):
        link_mat, new_score, end_score = self.w_link(objs, dets)

        if self.softmax_mode == 'single':
            link_score = F.softmax(link_mat, dim=-1)
        elif self.softmax_mode == 'dual':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = link_score_prev.mul(link_score_next)
        elif self.softmax_mode == 'dual_add':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = (link_score_prev + link_score_next) / 2
        elif self.softmax_mode == 'dual_max':
            link_score_prev = F.softmax(link_mat, dim=-1)
            link_score_next = F.softmax(link_mat, dim=-2)
            link_score = torch.max(link_score_prev, link_score_next)
        else:
            link_score = link_mat

        return link_score, new_score, end_score

    def generate_gt(self, det_score, det_cls, det_id, det_split):
        gt_det = det_score.new_zeros(det_score.size())
        gt_new = det_score.new_zeros(det_score.size())
        gt_end = det_score.new_zeros(det_score.size())
        gt_link = []
        det_start_idx = 0

        for i in range(len(det_split)):
            det_curr_num = det_split[i]  # current frame i has det_i detects
            if i != len(det_split) - 1:
                link_matrix = det_score.new_zeros(
                    (1, det_curr_num, det_split[i + 1]))
            # Assign the score, according to eq1
            for j in range(det_curr_num):
                curr_det_idx = det_start_idx + j
                # g_det
                if det_cls[i][j] == 1:
                    gt_det[curr_det_idx] = 1  # positive
                else:
                    continue

                # g_link for successor frame
                if i == len(det_split) - 1:
                    # end det at last frame
                    gt_end[curr_det_idx] = 1
                else:
                    matched = False
                    det_next_num = det_split[i + 1]
                    for k in range(det_next_num):
                        if det_id[i][j] == det_id[i + 1][k]:
                            link_matrix[0][j][k] = 1
                            matched = True
                            break
                    if not matched:
                        # no successor means an end det
                        gt_end[curr_det_idx] = 1

                if i == 0:
                    # new det at first frame
                    gt_new[curr_det_idx] = 1
                else:
                    # look prev
                    matched = False
                    det_prev_num = det_split[i - 1]
                    for k in range(det_prev_num):
                        if det_id[i][j] == det_id[i - 1][k]:
                            # have been matched during search in
                            # previous frame, no need to assign
                            matched = True
                            break
                    if not matched:
                        gt_new[curr_det_idx] = 1

            det_start_idx += det_curr_num
            if i != len(det_split) - 1:
                gt_link.append(link_matrix)

        return gt_det, gt_link, gt_new, gt_end

    def calculate_distance(self, dets, gt_dets):
        import motmetrics as mm
        distance = []
        # dets format: X1, Y1, X2, Y2
        # distance input format: X1, Y1, W, H
        # for i in range(len(dets)):
        det = dets.clone()
        det[:, 2:] = det[:, 2:] - det[:, :2]
        gt_det = gt_dets.clone()
        gt_det[:, 2:] = gt_det[:, 2:] - gt_det[:, :2]
        return mm.distances.iou_matrix(gt_det.cpu().detach().numpy(), det.cpu().detach().numpy(), max_iou=0.5)


    def generate_det_id_matrix_3d(self, dets_bbox, gt_dets, gt_ids, p_dets_bbox, p_gt_dets, p_gt_ids, c_class, p_class):
        distance = self.calculate_distance(dets_bbox[0][:,[0,1,3,4]], gt_dets[:,[0,1,3,4]])
        p_distance = self.calculate_distance(p_dets_bbox[0][:,[0,1,3,4]], p_gt_dets[0][:,[0,1,3,4]])
        mat = distance.copy()  # the smaller the value, the close between det and gt
        mat[np.isnan(mat)] = 10  # just set it to a big number
        p_mat = p_distance.copy()  # the smaller the value, the close between det and gt
        p_mat[np.isnan(p_mat)] = 10  # just set it to a big number
        v, idx = torch.min(torch.Tensor(mat), dim=-1)
        if p_mat.shape != (0,0):
            p_v, p_idx = torch.min(torch.Tensor(p_mat), dim=-1)
            gt_id = -1 * torch.ones((dets_bbox.shape[1], 1))
            gt_cls = torch.zeros((dets_bbox.shape[1], 1))
            p_gt_id = -1 * torch.ones((p_dets_bbox.shape[1], 1))
            p_gt_cls = torch.zeros((p_dets_bbox.shape[1], 1))
            for i in range(len(idx)):
                gt_id[idx[i]] = int(gt_ids[i])
                if c_class[0][i] == 'Car':
                    gt_cls[idx[i]] = 1
                elif c_class[0][i] == 'DontCare':
                    gt_cls[idx[i]] = -1
                else:
                    gt_cls[idx[i]] = 0
            for i in range(len(p_idx)):
                p_gt_id[p_idx[i]] = int(p_gt_ids[i])
                if p_class[0][i] == 'Car':
                    p_gt_cls[idx[i]] = 1
                elif p_class[0][i] == 'DontCare':
                    p_gt_cls[idx[i]] = -1
                else:
                    p_gt_cls[idx[i]] = 0
            gt_ids = [p_gt_id, gt_id]
            gt_clss = [p_gt_cls, gt_cls]
            k = 0
            return gt_ids, gt_clss, k
        else:
            gt_id = -1 * torch.ones((dets_bbox.shape[1], 1))
            gt_cls = torch.zeros((dets_bbox.shape[1], 1))
            p_gt_id = -1 * torch.ones((p_dets_bbox.shape[1], 1))
            p_gt_cls = torch.zeros((p_dets_bbox.shape[1], 1))
            for i in range(len(idx)):
                gt_id[idx[i]] = int(gt_ids[i])
                gt_cls[idx[i]] = 1
            # for i in range(len(p_idx)):
            #     p_gt_id[p_idx[i]] = int(p_gt_ids[i])
            #     p_gt_cls[p_idx[i]] = 1
            gt_ids = [p_gt_id, gt_id]
            gt_clss = [p_gt_cls, gt_cls]
            k = 1
            return gt_ids, gt_clss, k

    def generate_det_id_matrix(self, dets_bbox, gt_dets, gt_ids, p_dets_bbox, p_gt_dets, p_gt_ids, c_class, p_class):
        distance = self.calculate_distance(dets_bbox[0], gt_dets)
        p_distance = self.calculate_distance(p_dets_bbox[0], p_gt_dets)
        mat = distance.copy()  # the smaller the value, the close between det and gt
        mat[np.isnan(mat)] = 10  # just set it to a big number
        p_mat = p_distance.copy()  # the smaller the value, the close between det and gt
        p_mat[np.isnan(p_mat)] = 10  # just set it to a big number
        v, idx = torch.min(torch.Tensor(mat), dim=-1)
        if p_mat.shape != (0,0):
            p_v, p_idx = torch.min(torch.Tensor(p_mat), dim=-1)
            gt_id = -1 * torch.ones((dets_bbox.shape[1], 1))
            gt_cls = torch.zeros((dets_bbox.shape[1], 1))
            p_gt_id = -1 * torch.ones((p_dets_bbox.shape[1], 1))
            p_gt_cls = torch.zeros((p_dets_bbox.shape[1], 1))
            for i in range(len(idx)):
                gt_id[idx[i]] = int(gt_ids[i])
                if c_class[0][i] == 'Car':
                    gt_cls[idx[i]] = 1
                elif c_class[0][i] == 'DontCare':
                    gt_cls[idx[i]] = -1
                else:
                    gt_cls[idx[i]] = 0
            for i in range(len(p_idx)):
                p_gt_id[p_idx[i]] = int(p_gt_ids[i])
                if p_class[0][i] == 'Car':
                    p_gt_cls[p_idx[i]] = 1
                elif p_class[0][i] == 'DontCare':
                    p_gt_cls[p_idx[i]] = -1
                else:
                    p_gt_cls[p_idx[i]] = 0
            gt_ids = [p_gt_id, gt_id]
            gt_clss = [p_gt_cls, gt_cls]
            k = 0
            return gt_ids, gt_clss, k
        else:
            gt_id = -1 * torch.ones((dets_bbox.shape[1], 1))
            gt_cls = torch.zeros((dets_bbox.shape[1], 1))
            p_gt_id = -1 * torch.ones((p_dets_bbox.shape[1], 1))
            p_gt_cls = torch.zeros((p_dets_bbox.shape[1], 1))
            for i in range(len(idx)):
                gt_id[idx[i]] = int(gt_ids[i])
                if c_class[0][i] == 'Car':
                    gt_cls[idx[i]] = 1
                elif c_class[0][i] == 'DontCare':
                    gt_cls[idx[i]] = -1
                else:
                    gt_cls[idx[i]] = 0
            # for i in range(len(p_idx)):
            #     p_gt_id[p_idx[i]] = int(p_gt_ids[i])
            #     p_gt_cls[p_idx[i]] = 1
            gt_ids = [p_gt_id, gt_id]
            gt_clss = [p_gt_cls, gt_cls]
            k = 1
            return gt_ids, gt_clss, k

    def nms_vid(self, box_preds, cls_preds):
        iou_preds = torch.zeros_like(cls_preds)
        for box_pred, cls_pred, iou_pred in zip(box_preds, cls_preds, iou_preds):
            box_pred = box_pred.float()
            cls_pred = cls_pred.float()
            iou_pred = iou_pred.float()
            nms_func = box_torch_ops.rotate_nms
            total_scores = torch.sigmoid(cls_pred)
            total_ious = torch.sigmoid(iou_pred)
            top_scores = total_scores.squeeze(-1)
            top_ious = total_ious.squeeze(-1)
            top_labels = torch.zeros(
                total_scores.shape[0],
                device=total_scores.device,
                dtype=torch.long)
            top_scores_keep = top_scores >= 0.2
            top_scores = top_scores.masked_select(top_scores_keep)
            top_ious = top_scores.clone()            
            if top_scores.shape[0] != 0:
                box_pred = box_pred[top_scores_keep]
                # if self._use_direction_classifier:
                #     dir_labels = dir_labels[selected_idx[512*dir_idx:512*(dir_idx+1)]][top_scores_keep]
                top_labels = top_labels[top_scores_keep]
                boxes_for_nms = box_pred[:, [0, 1, 3, 4, 6]]

                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_ious,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                )
                selected_boxes = box_preds[:,selected]
            else:
                selected = torch.tensor([])
                selected_boxes = box_preds[:,[]]

        return selected_boxes, selected

    def top_to_img(self, batch_c_box_preds, example, center_limit_range):
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        out_idx = 0
        center_limit_range = torch.tensor([0.0, -40.0, -3.0, 70.4, 40.0, 0.0]).cuda()

        det_imgs = []

        # import cv2
        # img = example['f_view'][0]
        # f_view = img.detach().cpu().numpy()
        # f_view = f_view.transpose(1,2,0)
        # f_mean = np.array([0.485,0.456,0.406])
        # f_std = np.array([0.229,0.224,0.225])
        # f_view = ((f_view * f_std) + f_mean) * 255.
        # f_view = f_view.astype(np.uint8)


        for box_preds, rect, Trv2c, P2, image_shape, img in zip(batch_c_box_preds,  batch_rect, batch_Trv2c, batch_P2, example['image_shape'], example['f_view']):
            box_preds = box_preds.float()
            rect = rect.float()
            Trv2c = Trv2c.float()
            P2 = P2.float()
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)
            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            xmin_idx = box_2d_preds[:,0] > image_shape[1]
            ymin_idx = box_2d_preds[:,1] > image_shape[0]
            xmax_idx = box_2d_preds[:,2] < 0
            ymax_idx = box_2d_preds[:,3] < 0
            # xl_idx = (box_preds[:,:3] < center_limit_range[:3]).sum(dim=1)
            # yl_idx = (box_preds[:,:3] > center_limit_range[3:]).sum(dim=1)
            # for i in range(box_2d_preds.shape[0]):
            #     out_idx += 1
            #     if box_2d_preds[i][0] > image_shape[1] or box_2d_preds[i][1] > image_shape[0]:
            #         continue
            #     if box_2d_preds[i][2] < 0 or box_2d_preds[i][3] < 0:
            #         continue
                # if (np.any(box_lidar[:3] < limit_range[:3])
                #     or np.any(box_lidar[:3] > limit_range[3:])):
                #     continue
            box_2d_preds[:,2:] = torch.min(box_2d_preds[:,2:], torch.tensor([image_shape[1], image_shape[0]]).cuda().float())
            box_2d_preds[:,:2] = torch.max(box_2d_preds[:,:2], torch.tensor([0., 0.]).cuda())
            xsame_idx = (box_2d_preds[:,0]).to(torch.int) == (box_2d_preds[:,2]).to(torch.int)
            ysame_idx = (box_2d_preds[:,1]).to(torch.int) == (box_2d_preds[:,3]).to(torch.int)
            # idxs = xmin_idx + ymin_idx + xmax_idx + ymax_idx + xl_idx.to(torch.uint8) + yl_idx.to(torch.uint8) + xsame_idx + ysame_idx
            idxs = xmin_idx + ymin_idx + xmax_idx + ymax_idx + xsame_idx + ysame_idx
            out_idxs = idxs ==0

            for i in range(box_2d_preds[out_idxs].shape[0]):
                x1 = int(box_2d_preds[out_idxs][i, 0])
                y1 = int(box_2d_preds[out_idxs][i, 1])
                x2 = int(box_2d_preds[out_idxs][i, 2])
                y2 = int(box_2d_preds[out_idxs][i, 3])
                det_imgs.append(F.upsample_bilinear(img[:,y1:y2,x1:x2].unsqueeze(0), [224,224]))
                # f_view = cv2.rectangle(f_view, (x1,y1), (x2,y2), (0,0,255), 2)

            # cv2.imwrite('test.jpg', f_view)
            try:
                det_imgs = torch.cat(det_imgs, dim=0)
            except:
                det_imgs = torch.zeros(0)

        return det_imgs, out_idxs, box_2d_preds[out_idxs]

    def p_top_to_img(self, batch_c_box_preds, example, center_limit_range):
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        out_idx = 0
        # center_limit_range = torch.tensor([0.0, -40.0, -3.0, 70.4000015258789, 40.0, 0.0]).cuda()
        det_imgs = []
        for box_preds, rect, Trv2c, P2, image_shape, img in zip(batch_c_box_preds,  batch_rect, batch_Trv2c, batch_P2, example['p_image_shape'], example['p_f_view']):
            box_preds = box_preds.float()
            rect = rect.float()
            Trv2c = Trv2c.float()
            P2 = P2.float()
            final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    box_preds, rect, Trv2c)
            locs = final_box_preds_camera[:, :3]
            dims = final_box_preds_camera[:, 3:6]
            angles = final_box_preds_camera[:, 6]
            camera_box_origin = [0.5, 1.0, 0.5]
            box_corners = box_torch_ops.center_to_corner_box3d(
                locs, dims, angles, camera_box_origin, axis=1)
            box_corners_in_image = box_torch_ops.project_to_image(
                box_corners, P2)
            minxy = torch.min(box_corners_in_image, dim=1)[0]
            maxxy = torch.max(box_corners_in_image, dim=1)[0]
            box_2d_preds = torch.cat([minxy, maxxy], dim=1)
            xmin_idx = box_2d_preds[:,0] > image_shape[1]
            ymin_idx = box_2d_preds[:,1] > image_shape[0]
            xmax_idx = box_2d_preds[:,2] < 0
            ymax_idx = box_2d_preds[:,3] < 0
            # xl_idx = (box_preds[:,:3] < center_limit_range[:3]).sum(dim=1)
            # yl_idx = (box_preds[:,:3] > center_limit_range[3:]).sum(dim=1)
            # for i in range(box_2d_preds.shape[0]):
            #     out_idx += 1
            #     if box_2d_preds[i][0] > image_shape[1] or box_2d_preds[i][1] > image_shape[0]:
            #         continue
            #     if box_2d_preds[i][2] < 0 or box_2d_preds[i][3] < 0:
            #         continue
                # if (np.any(box_lidar[:3] < limit_range[:3])
                #     or np.any(box_lidar[:3] > limit_range[3:])):
                #     continue
            box_2d_preds[:,2:] = torch.min(box_2d_preds[:,2:], torch.tensor([image_shape[1], image_shape[0]]).cuda().float())
            box_2d_preds[:,:2] = torch.max(box_2d_preds[:,:2], torch.tensor([0., 0.]).cuda())
            xsame_idx = (box_2d_preds[:,0]).to(torch.int) == (box_2d_preds[:,2]).to(torch.int)
            ysame_idx = (box_2d_preds[:,1]).to(torch.int) == (box_2d_preds[:,3]).to(torch.int)
            # idxs = xmin_idx + ymin_idx + xmax_idx + ymax_idx + xl_idx.to(torch.uint8) + yl_idx.to(torch.uint8) + xsame_idx + ysame_idx
            idxs = xmin_idx + ymin_idx + xmax_idx + ymax_idx + xsame_idx + ysame_idx
            out_idxs = idxs ==0

            for i in range(box_2d_preds[out_idxs].shape[0]):
                x1 = int(box_2d_preds[out_idxs][i, 0])
                y1 = int(box_2d_preds[out_idxs][i, 1])
                x2 = int(box_2d_preds[out_idxs][i, 2])
                y2 = int(box_2d_preds[out_idxs][i, 3])
                # x1 = np.floor(box_2d_preds[out_idxs][i, 0])
                # y1 = np.floor(box_2d_preds[out_idxs][i, 1])
                # x2 = np.ceil(box_2d_preds[out_idxs][i, 2])
                # y2 = np.ceil(box_2d_preds[out_idxs][i, 3])
                # dim.append(frame['detection']['dimensions'][i:i+1])
                # loc.append(location[i:i+1])
                # rot.append(rotation_y[i:i+1].reshape(1,1))
                det_imgs.append(F.upsample_bilinear(img[:,y1:y2,x1:x2].unsqueeze(0), [224,224]))
            
            try:
                det_imgs = torch.cat(det_imgs, dim=0)
            except:
                det_imgs = torch.zeros(0)

        return det_imgs, out_idxs, box_2d_preds[out_idxs]

    def remove_points_outside_boxes(self, points, boxes):
        masks = points_in_rbbox(points, boxes)
        points = points[masks.any(-1)]
        return points

    def remove_outside_points(self, points, rect, Trv2c, P2, image_shape):
        # 5x faster than remove_outside_points_v1(2ms vs 10ms)
        C, R, T = projection_matrix_to_CRT_kitti(P2)
        image_bbox = [0, 0, image_shape[1], image_shape[0]]
        frustum = get_frustum(image_bbox, C)
        frustum -= T
        frustum = np.linalg.inv(R) @ frustum.T
        frustum = camera_to_lidar(frustum.T, rect, Trv2c)
        frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
        indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
        points = points[indices.reshape([-1])]
        return points

def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                iou_loss_ftor=None,
                iou_preds=None,
                iou_targets=None,
                use_iou_loss=False,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    if use_iou_loss:
        iou_losses = iou_loss_ftor(
            iou_preds, iou_targets, weights=cls_weights)
    else:
        iou_losses = 0.
    return loc_losses, cls_losses, iou_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets

