
export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib

# save train script
WAND_NAME="h1_v58"
CONFIG_BACKUP_DIR="/mnt/hypercube/zhsha/workspace/pbrs-humanoid/config_bakcup/${WAND_NAME}"
mkdir -p $CONFIG_BACKUP_DIR
script_path="$(readlink -f "$0")"
cp $script_path $CONFIG_BACKUP_DIR

python gpugym/scripts/train.py \
    --task pbrs:H1 \
    --headless \
    --wandb_project "pbrs3" \
    --wandb_name $WAND_NAME \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 1.0 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 20.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0
