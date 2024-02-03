
export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib
export DISPLAY=localhost:10.0

# h1 robot
python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v35 \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 1.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.0 \
    --ang_vel_yaw_abs 0.0

# # mit robot
# python gpugym/scripts/play.py \
#     --task pbrs:humanoid \
#     --wandb_name h1_v35 \
#     --h1_urdf_version 8 \
#     --action_scale 0.5 \
#     --ori_term_threshold 1.0 \
#     --lin_vel_x_min 0.0 \
#     --lin_vel_x_max 4.5 \
#     --lin_vel_y_ab 0.0 \
#     --ang_vel_yaw_abs 0.0

