# _model_name="autocal_perf"
# _input_config="onestage/autocal_server_fusion_70e.fhd.onestage.config"
# _use_fusion=True
# _use_second=False
# _use_endtoend=False

# LOOP_O="1"
# # LOOP_O="1"
# for c in $LOOP_O
# do
#     CUDA_VISIBLE_DEVICES=$1 python pytorch/train.py train\
#     --config_path="configs/${_input_config}"\
#     --model_dir="logs/${_model_name}_gpu$1_$c"\
#     --use_fusion=${_use_fusion}\
#     --use_second=${_use_second}\
#     --use_endtoend=${_use_endtoend}
# done


# CUDA_VISIBLE_DEVICES=2 python pytorch/train.py train --config_path="configs/twostage_iou/vis_server_2st_fusion.fhd.onestage.config" --model_dir="logs/runtest" --use_fusion True --use_endtoend=True

CUDA_VISIBLE_DEVICES=2 python pytorch/train.py train --config_path="configs/onestage/autocal_server_fusion_50e.fhd.onestage.config" --model_dir="logs/rgb_test" --use_fusion True

CUDA_VISIBLE_DEVICES=2 python pytorch/train.py evaluate --config_path="configs/onestage/autocal_server_fusion_50e.fhd.onestage.config" --model_dir="logs/depth_test" --use_fusion True --ckpt_path="logs/depth_test/voxelnet-39691.tckpt"   

# CUDA_VISIBLE_DEVICES=1 python pytorch/train.py evaluate --config_path="configs/twostage_iou/demo_server_2st_fusion_test.fhd.onestage.config" --model_dir="logs/end-to-end-mod_lr2_gpu1_2" --ckpt_path="logs/end-to-end-mod_lr2_gpu1_2/voxelnet-4024.tckpt"   --use_fusion True --use_endtoend=True 