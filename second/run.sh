# rsync -avz -e 'ssh -p 4799' --exclude='dataset' --exclude="logs*" --exclude='prev_logs' --exclude="*.pyc" --exclude='*.tckpt' --exclude='*.png' jhyoo@166.104.14.92:/home/jhyoo/jhyoo/3DOD/iou_15_second/new_iou/ /home/spalab/jskim/kitti_15_second/
# rsync -avz -e 'ssh -p 4799' --exclude='dataset' --exclude="logs*" --exclude='prev_logs' --exclude="*.pyc" --exclude='*.tckpt' --exclude='*.png' jhyoo@166.104.14.92:/home/jhyoo/jhyoo/3DOD/iou_15_second/new_iou/ /home/spalab/jhyoo/new_15_second/
python create_data.py create_kitti_info_file --data_path=/home/spalab/jhyoo/new_15_second/second.pytorch/second/dataset/
python create_data.py create_reduced_point_cloud --data_path=/home/spalab/jhyoo/new_15_second/second.pytorch/second/dataset/
python create_data.py create_groundtruth_database --data_path=/home/spalab/jhyoo/new_15_second/second.pytorch/second/dataset/
