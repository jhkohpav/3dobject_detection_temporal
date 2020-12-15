import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import json 
import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter
import torchvision
import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess_tr import merge_second_batch_tr
from second.data.preprocess import merge_second_batch
from second.data.preprocess_tr_vid import merge_second_batch_tr_vid
from second.data.preprocess_tr_vid_spatio import merge_second_batch_tr_vid_spatio
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder_tr, input_reader_builder_tr_vid, input_reader_builder_tr_vid_spatio,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder,
                                      second_2stage_builder,
                                      second_endtoend_builder,
                                      second_endtoend_builder_tr,
                                      second_endtoend_builder_tr_share,
                                      second_endtoend_builder_tr_share_freeze,
                                      second_endtoend_builder_tr_share_freeze_mmmot,
                                      second_endtoend_builder_tr_share_freeze_mmmot_ori,
                                      second_endtoend_builder_spatio)
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import ProgressBar
from collections import OrderedDict
# import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP

import sys
sys.path.append('./mmMOT')
import argparse
import logging
import os
import pprint
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import yaml
from easydict import EasyDict
from kitti_devkit.evaluate_tracking import evaluate as evaluate_tr
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from models import model_entry
# from tracking_model import TrackingModule
from tracking_model_vid import TrackingModule
from utils_tr.build_util import (build_augmentation, build_criterion, 
                              build_dataset, build_lr_scheduler, build_model,
                              build_optim)
from utils_tr.data_util import write_kitti_result
from utils_tr.train_util import (AverageMeter, DistributedGivenIterationSampler,
                              create_logger, load_state, save_checkpoint)


def validate(val_loader,
             net,
             step,
             config,
             result_path,
             part='train',
             fusion_list=None,
             fuse_prob=False):

    logger = logging.getLogger('global_logger')
    for i, (sequence) in enumerate(val_loader):
        logger.info('Test: [{}/{}]\tSequence ID: KITTI-{}'.format(
            i, len(val_loader), sequence.name))
        seq_loader = DataLoader(
            sequence,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True)
        if len(seq_loader) == 0:
            net.eval_tr()
            logger.info('Empty Sequence ID: KITTI-{}, skip'.format(
                sequence.name))
        else:
            validate_seq(seq_loader, net, config)

        write_kitti_result(
            result_path,
            sequence.name,
            step,
            net.frames_id,
            net.frames_det,
            part=part)

    MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = evaluate_tr(
        step, result_path, part=part)

    # net.train()
    return MOTA, MOTP, recall, prec, F1, fp, fn, id_switches


def validate_seq(val_loader,
                 net,
                 config,
                 fusion_list=None,
                 fuse_prob=False):
    batch_time = AverageMeter(0)

    # switch to evaluate mode
    net.eval_tr()

    logger = logging.getLogger('global_logger')
    end = time.time()

    with torch.no_grad():
        for i, (input, det_info, dets, det_split) in enumerate(val_loader):
            input = input.cuda()
            if len(det_info) > 0:
                for k, v in det_info.items():
                    det_info[k] = det_info[k].cuda() if not isinstance(
                        det_info[k], list) else det_info[k]

            # compute output
            aligned_ids, aligned_dets, frame_start = net.predict(
                input[0], det_info, dets, det_split)

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.print_freq == 0:
                logger.info(
                    'Test Frame: [{0}/{1}]\tTime '
                    '{batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time))


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


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2", "f_view","idxs_norm", "p_voxels", "p_f_view", "p_idxs_norm", 'box_id', 'p_box_id', 'gt_boxes', 'p_gt_boxes', 'boxes_2d', 'p_boxes_2d'
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.tensor(v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points", "p_coordinates", "p_num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch


def train(config_path,
          model_dir,
          use_fusion=True,
          use_ft=False,
          use_second_stage=True,
          use_endtoend=True,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,
          local_rank=0,
          pickle_result=True,
          patchs=None):
    """train a VoxelNet mod[el specified by a config file.
    """
    ############ tracking
    config_tr_path = '/mnt/new_iou/second.pytorch/second/mmMOT/experiments/second/spatio_test/config.yaml'
    load_tr_path = '/mnt/new_iou/second.pytorch/second/mmMOT/experiments/second/spatio_test/results'
    with open(config_tr_path) as f:
        config_tr = yaml.load(f, Loader=yaml.FullLoader)

    result_path_tr = load_tr_path
    config_tr = EasyDict(config_tr['common'])
    config_tr.save_path = os.path.dirname(config_tr_path)

    # create model
    # model_tr = build_model(config_tr)
    # model_tr.cuda()

    # optimizer_tr = build_optim(model_tr, config_tr)

    criterion_tr = build_criterion(config_tr.loss)

    last_iter = -1
    best_mota = 0
    # if load_tr_path:
    #     if False:
    #         best_mota, last_iter = load_state(
    #             load_tr_path, model_tr, optimizer=optimizer_tr)
    #     else:
    #         load_state(load_tr_path, model_tr)

    cudnn.benchmark = True

    # Data loading code
    train_transform, valid_transform = build_augmentation(config_tr.augmentation)

    # # train
    # train_dataset = build_dataset(
    #     config_tr,
    #     set_source='train',
    #     evaluate=False,
    #     train_transform=train_transform)
    # trainval_dataset = build_dataset(
    #     config_tr,
    #     set_source='train',
    #     evaluate=True,
    #     valid_transform=valid_transform)
    # val_dataset = build_dataset(
    #     config_tr,
    #     set_source='val',
    #     evaluate=True,
    #     valid_transform=valid_transform)

    # train_sampler = DistributedGivenIterationSampler(
    #     train_dataset,
    #     config_tr.lr_scheduler.max_iter,
    #     config_tr.batch_size,
    #     world_size=1,
    #     rank=0,
    #     last_iter=last_iter)

    # import pdb; pdb.set_trace()
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config_tr.batch_size,
    #     shuffle=False,
    #     num_workers=config_tr.workers,
    #     pin_memory=True)

    tb_logger = SummaryWriter(config_tr.save_path + '/events')
    logger = create_logger('global_logger', config_tr.save_path + '/log.txt')
    # logger.info('args: {}'.format(pprint.pformat(args)))
    logger.info('config: {}'.format(pprint.pformat(config_tr)))

    # tracking_module = TrackingModule(model_tr, criterion_tr,
                                    #  config_tr.det_type)
    # tracking_module.model.train()
    #### tracking setup done

    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    patchs = patchs or []
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    for patch in patchs:
        patch = "config." + patch 
        exec(patch)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    class_names = target_assigner.classes
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    # if use_second_stage:
    #     net = second_2stage_builder.build(model_cfg, voxel_generator, target_assigner)
    if use_endtoend:
        net = second_endtoend_builder_spatio.build(model_cfg, voxel_generator, target_assigner, criterion_tr, config_tr.det_type)
    else:
        net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net.cuda()
    print("num_trainable parameters:", len(list(net.parameters())))

    for n, p in net.named_parameters():
        print(n, p.shape)
    # pth_name = './pre_weight/first_stage_gating_det/voxelnet-17013.tckpt'
    pth_name = './pre_weight/second_stage_gating_det/voxelnet-35000.tckpt'

    res_pre_weights = torch.load(pth_name)
    new_res_state_dict = OrderedDict()
    model_dict = net.state_dict()
    for k,v in res_pre_weights.items():
        if 'global_step' not in k:
            # if 'dir' not in k:
            new_res_state_dict[k] = v
    model_dict.update(new_res_state_dict)
    net.load_state_dict(model_dict)

    # for k, weight in dict(net.named_parameters()).items(): # lidar_conv, p_lidar_conv, fusion_module, w_det, w_link, appearance, point_net
    #     if 'middle_feature_extractor' in '%s'%(k) or 'rpn' in '%s'%(k) or 'second_rpn' in '%s'%(k):
    #         weight.requires_grad = False

    # BUILD OPTIMIZER
    #####################
    # we need global_step to create lr_scheduler, so restore net first.
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    loss_scale = train_cfg.loss_scale_factor
    mixed_optimizer = optimizer_builder.build(optimizer_cfg, net, mixed=train_cfg.enable_mixed_precision, loss_scale=loss_scale)
    optimizer = mixed_optimizer

    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [mixed_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, train_cfg.steps)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    ######################
    # PREPARE INPUT
    ######################
    # import pdb; pdb.set_trace()
    dataset = input_reader_builder_tr_vid_spatio.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        config_tr=config_tr,
        set_source='train',
        evaluate=False,
        train_transform=train_transform)
    eval_dataset = input_reader_builder_tr_vid_spatio.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        config_tr=config_tr,
        set_source='val',
        evaluate=True,
        valid_transform=valid_transform)

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch_tr_vid_spatio,
        worker_init_fn=_worker_init_fn)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch_tr_vid_spatio)
    
    data_iter = iter(dataloader)

    ######################
    # TRAINING
    ######################
    training_detail = []
    log_path = model_dir / 'log.txt'
    training_detail_path = model_dir / 'log.json'
    if training_detail_path.exists():
        with open(training_detail_path, 'r') as f:
            training_detail = json.load(f)
    logf = open(log_path, 'a')
    logf.write(proto_str)
    logf.write("\n")
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()
    
    # optimizer_tr.zero_grad()
    logger = logging.getLogger('global_logger')
    best_mota = 0
    losses = AverageMeter(config_tr.print_freq)

    total_steps = train_cfg.steps
    total_loop = total_steps // len(dataloader)
    
    kkkk = 0
    for step in range(total_loop):
        for i, (example) in enumerate(dataloader):

            curr_step = 0 + i
            kkkk += 1
            lr_scheduler.step(net.get_global_step())

            example_torch = example_convert_to_torch(example, float_dtype)

            batch_size = example["anchors"].shape[0]

            ret_dict = net(example_torch, train_param=True)

            cls_preds = ret_dict["cls_preds"]
            loss = ret_dict["loss"].mean()
            cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
            loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
            cls_pos_loss = ret_dict["cls_pos_loss"]
            cls_neg_loss = ret_dict["cls_neg_loss"]
            loc_loss = ret_dict["loc_loss"]
            cls_loss = ret_dict["cls_loss"]
            dir_loss_reduced = ret_dict["dir_loss_reduced"]
            cared = ret_dict["cared"]
            # loss_tr = ret_dict["loss_tr"]

            if use_second_stage or use_endtoend:
                labels = ret_dict["labels"]
            else:
                labels = example_torch["labels"]
            if train_cfg.enable_mixed_precision:
                loss *= loss_scale

            try:
                loss.backward()
            except:
                abc = 1
            #     import pdb; pdb.set_trace()
            #     abc = 1
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            # optimizer_tr.step()
            # optimizer_tr.zero_grad()
            mixed_optimizer.step()
            mixed_optimizer.zero_grad()
            net.update_global_step()
            net_metrics = net.update_metrics(cls_loss_reduced,
                                                loc_loss_reduced, cls_preds,
                                                labels, cared)

            step_time = (time.time() - t)
            t = time.time()
            metrics = {}
            num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
            num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
            if 'anchors_mask' not in example_torch:
                num_anchors = example_torch['anchors'].shape[1]
            else:
                num_anchors = int(example_torch['anchors_mask'][0].sum())
            global_step = net.get_global_step()
            # print(step)
            if global_step % display_step == 0:
                loc_loss_elem = [
                    float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                            batch_size) for i in range(loc_loss.shape[-1])
                ]
                metrics["type"] = "step_info"
                metrics["step"] = global_step
                metrics["steptime"] = step_time
                metrics.update(net_metrics)
                metrics["loss"] = {}
                metrics["loss"]["loc_elem"] = loc_loss_elem
                metrics["loss"]["cls_pos_rt"] = float(
                    cls_pos_loss.detach().cpu().numpy())
                metrics["loss"]["cls_neg_rt"] = float(
                    cls_neg_loss.detach().cpu().numpy())
                if model_cfg.use_direction_classifier:
                    metrics["loss"]["dir_rt"] = float(
                        dir_loss_reduced.detach().cpu().numpy())
                metrics["num_vox"] = int(example_torch["voxels"].shape[0])
                metrics["num_pos"] = int(num_pos)
                metrics["num_neg"] = int(num_neg)
                metrics["num_anchors"] = int(num_anchors)
                metrics["lr"] = float(
                    optimizer.lr)

                metrics["image_idx"] = example['image_idx'][0][7:]
                training_detail.append(metrics)
                flatted_metrics = flat_nested_json_dict(metrics)
                flatted_summarys = flat_nested_json_dict(metrics, "/")
                for k, v in flatted_summarys.items():
                    if isinstance(v, (list, tuple)):
                        v = {str(i): e for i, e in enumerate(v)}
                        if type(v) != str and ('loc_elem' not in k):
                            writer.add_scalars(k, v, global_step)
                    else:
                        if (type(v) != str) and ('loc_elem' not in k):
                            writer.add_scalar(k, v, global_step)

                metrics_str_list = []
                for k, v in flatted_metrics.items():
                    if isinstance(v, float):
                        metrics_str_list.append(f"{k}={v:.3}")
                    elif isinstance(v, (list, tuple)):
                        if v and isinstance(v[0], float):
                            v_str = ', '.join([f"{e:.3}" for e in v])
                            metrics_str_list.append(f"{k}=[{v_str}]")
                        else:
                            metrics_str_list.append(f"{k}={v}")
                    else:
                        metrics_str_list.append(f"{k}={v}")
                log_str = ', '.join(metrics_str_list)
                print(log_str, file=logf)
                print(log_str)

            ckpt_elasped_time = time.time() - ckpt_start_time
            if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())

                ckpt_start_time = time.time()

            if kkkk > 0 and (kkkk) % config_tr.val_freq == 0:
            # if True:
                torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())
                net.eval()
                result_path_step = result_path / f"step_{net.get_global_step()}"
                result_path_step.mkdir(parents=True, exist_ok=True)
                print("#################################")
                print("#################################", file=logf)
                print("# EVAL")
                print("# EVAL", file=logf)
                print("#################################")
                print("#################################", file=logf)
                print("Generate output labels...")
                print("Generate output labels...", file=logf)
                t = time.time()
                dt_annos = []
                prog_bar = ProgressBar()
                net.clear_timer()
                prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1) // eval_input_cfg.batch_size)
                for example in iter(eval_dataloader):
                    example = example_convert_to_torch(example, float_dtype)
                    if pickle_result:
                        results = predict_kitti_to_anno(
                            net, example, class_names, center_limit_range,
                            model_cfg.lidar_input)
                        dt_annos += results

                    else:
                        _predict_kitti_to_file(net, example, result_path_step,
                                            class_names, center_limit_range,
                                            model_cfg.lidar_input)

                    prog_bar.print_bar()

                sec_per_ex = len(eval_dataset) / (time.time() - t)
                print(f'generate label finished({sec_per_ex:.2f}/s). start eval:')
                print(f'generate label finished({sec_per_ex:.2f}/s). start eval:',file=logf)
                gt_annos = [
                    info["annos"] for info in eval_dataset.dataset.kitti_infos
                ]
                if not pickle_result:
                    dt_annos = kitti.get_label_annos(result_path_step)
                # result = get_official_eval_result_v2(gt_annos, dt_annos, class_names)
                # print(json.dumps(result, indent=2), file=logf)
                result = get_official_eval_result(gt_annos, dt_annos, class_names)
                print(result, file=logf)
                print(result)
                result_1 = result.split("\n")[:5]
                result_2 = result.split("\n")[10:15]
                result_3 = result.split("\n")[20:25]
                emh = ['0_easy', '1_mod', '2_hard']
                result_save = result_1
                for i in range(len(result_save)-1):
                    save_targ = result_save[i+1]
                    name_val = save_targ.split(':')[0].split(' ')[0]
                    value_val = save_targ.split(':')[1:]
                    for ev in range(3):
                        each_val = value_val[0].split(',')[ev]
                        merge_txt = 'AP_kitti/car_70/' + name_val+'/'+emh[ev]
                        try:
                            writer.add_scalar(merge_txt, float(each_val), global_step)
                        except:
                            abc=1
                            import pdb; pdb.set_trace()
                            abc=1
                if pickle_result:
                    with open(result_path_step / "result.pkl", 'wb') as f:
                        pickle.dump(dt_annos, f)
                writer.add_text('eval_result', result, global_step)

                logger.info('Evaluation on validation set:')
                # MOTA, MOTP, recall, prec, F1, fp, fn, id_switches = validate(
                #     val_dataset,
                #     net,
                #     str(0 + 1),
                #     config_tr,
                #     result_path_tr,
                #     part='val')
                # print(MOTA, MOTP, recall, prec, F1, fp, fn, id_switches)

                # curr_step = step
                # if tb_logger is not None:
                #     tb_logger.add_scalar('prec', prec, curr_step)
                #     tb_logger.add_scalar('recall', recall, curr_step)
                #     tb_logger.add_scalar('mota', MOTA, curr_step)
                #     tb_logger.add_scalar('motp', MOTP, curr_step)
                #     tb_logger.add_scalar('fp', fp, curr_step)
                #     tb_logger.add_scalar('fn', fn, curr_step)
                #     tb_logger.add_scalar('f1', F1, curr_step)
                #     tb_logger.add_scalar('id_switches', id_switches, curr_step)
                    # if lr_scheduler is not None:
                        # tb_logger.add_scalar('lr', current_lr, curr_step)

                # is_best = MOTA > best_mota
                # best_mota = max(MOTA, best_mota)
                # print(best_mota)

                # import pdb; pdb.set_trace()
                # save_checkpoint(
                #     {   'step': net.get_global_step(),
                #         'score_arch': config_tr.model.score_arch,
                #         'appear_arch': config_tr.model.appear_arch,
                #         'best_mota': best_mota,
                #         'state_dict': tracking_module.model.state_dict(),
                #         'optimizer': tracking_module.optimizer.state_dict(),
                #     }, is_best, config_tr.save_path + '/ckpt')

                # net.train()

    # save model before exit
    torchplus.train.save_models(model_dir, [net, optimizer],
                                net.get_global_step())
    logf.close()


def _predict_kitti_to_file(net,
                           example,
                           result_save_path,
                           class_names,
                           center_limit_range=None,
                           lidar_input=False):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts, assign_det, assign_link, assign_new, assign_end = net(example)
    # t = time.time()
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"][7:]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel():
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[:, [0, 1, 2, 4, 5, 3,
                                      6]]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    'name': class_names[int(label)],
                    'alpha': -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    'bbox': bbox,
                    'location': box[:3],
                    'dimensions': box[3:6],
                    'rotation_y': box[6],
                    'score': score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        result_str = '\n'.join(result_lines)
        with open(result_file, 'w') as f:
            f.write(result_str)


def predict_kitti_to_anno(net,
                          example,
                          class_names,
                          center_limit_range=None,
                          lidar_input=False,
                          global_set=None):
    batch_image_shape = example['image_shape']
    batch_imgidx = example['image_idx']
    predictions_dicts = net(example, False)
    # t = time.time()
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"][7:]
        if preds_dict["bbox"] is not None or preds_dict["bbox"].size.numel() != 0:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            # label_preds = np.zeros([box_2d_preds.shape[0]], dtype=np.int32)
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                    box_preds, box_preds_lidar, box_2d_preds, scores,
                    label_preds):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                # print(img_shape)
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if (np.any(box_lidar[:3] < limit_range[:3])
                            or np.any(box_lidar[:3] > limit_range[3:])):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(-np.arctan2(-box_lidar[1], box_lidar[0]) +
                                     box[6])
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        # import pdb; pdb.set_trace()
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64)
    return annos

def evaluate(config_path,
             model_dir,
             use_second_stage=False,
             use_endtoend=False,
             result_path=None,
             predict_test=False,
             ckpt_path=None,
             ref_detfile=None,
             pickle_result=True,
             measure_time=False,
             batch_size=None):
    model_dir = pathlib.Path(model_dir)
    if predict_test:
        result_name = 'predict_test_0095'
    else:
        result_name = 'eval_results'
    if result_path is None:
        result_path = model_dir / result_name
    else:
        result_path = pathlib.Path(result_path)
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config
    
    center_limit_range = model_cfg.post_center_limit_range
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    class_names = target_assigner.classes
    if use_second_stage:    
        net = second_2stage_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    elif use_endtoend:
        net = second_endtoend_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    else:
        net = second_builder.build(model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    net.cuda()
    #########################################
    # net = torch.nn.DataParallel(net)
    #########################################
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder_tr.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,# input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dt_annos = []
    global_set = None
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()
    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            t1 = time.time()
            torch.cuda.synchronize()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)

        if pickle_result:
            dt_annos += predict_kitti_to_anno(
                net, example, class_names, center_limit_range,
                model_cfg.lidar_input, global_set)
        else:
            _predict_kitti_to_file(net, example, result_path_step, class_names,
                                   center_limit_range, model_cfg.lidar_input)
        # print(json.dumps(net.middle_feature_extractor.middle_conv.sparity_dict))
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms")
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    if not predict_test:
        gt_annos = [info["annos"] for info in eval_dataset.dataset.kitti_infos]
        img_idx = [info["image_idx"] for info in eval_dataset.dataset.kitti_infos]
        if not pickle_result:
            dt_annos = kitti.get_label_annos(result_path_step)
        result = get_official_eval_result(gt_annos, dt_annos, class_names)
        # print(json.dumps(result, indent=2))
        print(result)
        result = get_coco_eval_result(gt_annos, dt_annos, class_names)
        print(result)
        if pickle_result:
            with open(result_path_step / "result.pkl", 'wb') as f:
                pickle.dump(dt_annos, f)
        # annos to txt file
        if True:
            os.makedirs(str(result_path_step) + '/txt', exist_ok=True)
            for i in range(len(dt_annos)):
                dt_annos[i]['dimensions'] = dt_annos[i]['dimensions'][:, [1, 2, 0]]
                result_lines = kitti.annos_to_kitti_label(dt_annos[i])
                image_idx = img_idx[i]
                with open(str(result_path_step) + '/txt/%06d.txt' % image_idx, 'w') as f:
                    for result_line in result_lines:
                        f.write(result_line + '\n')
                abcd = 1
    else:
        os.makedirs(str(result_path_step) + '/txt', exist_ok=True)
        img_idx = [info["image_idx"] for info in eval_dataset.dataset.kitti_infos]
        for i in range(len(dt_annos)):
            dt_annos[i]['dimensions'] = dt_annos[i]['dimensions'][:, [1, 2, 0]]
            result_lines = kitti.annos_to_kitti_label(dt_annos[i])
            image_idx = img_idx[i]
            with open(str(result_path_step) + '/txt/%06d.txt' % image_idx, 'w') as f:
                for result_line in result_lines:
                    f.write(result_line + '\n')


def save_config(config_path, save_path):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    ret = text_format.MessageToString(config, indent=2)
    with open(save_path, 'w') as f:
        f.write(ret)

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
                det_out['name'].append(dets[i]['name'][:, j])
                det_out['truncated'].append(dets[i]['truncated'][:, j])
                det_out['occluded'].append(dets[i]['occluded'][:, j])
                det_out['alpha'].append(dets[i]['alpha'][:, j])
                det_out['bbox'].append(dets[i]['bbox'][:, j])
                det_out['dimensions'].append(dets[i]['dimensions'][:, j])
                det_out['location'].append(dets[i]['location'][:, j])
                det_out['rotation_y'].append(dets[i]['rotation_y'][:, j])

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
        det_out['frame_idx'] = dets[i]['frame_idx']
        dets_out.append(det_out)
        det_start_idx += det_curr_num
    return det_ids, dets_out

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

if __name__ == '__main__':
    fire.Fire()
