
export CUDA_VISIBLE_DEVICES=1

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib

# H1 debug mode
python gpugym/scripts/train.py \
    --task pbrs:H1 \
    --headless \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 1.0 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --knee_stiffness 30.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 5.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0


# # MIT debug mode
# python gpugym/scripts/train.py \
#     --headless \
#     --task pbrs:humanoid 

    

