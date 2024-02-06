
export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib
export DISPLAY=localhost:10.0

# h1_v14
python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v14 \
    --h1_urdf_version 8 \
    --action_scale 1.0 \
    --ori_term_threshold 1.0 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --knee_stiffness 60.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 5.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0 \
    --knee_range_low 0.6 \
    --knee_range_high 0.7 

# h1_v35
python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v35 \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 1.0 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --knee_stiffness 60.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 5.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0 \
    --knee_range_low 0.6 \
    --knee_range_high 0.7 

# h1_v76
python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v76 \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 1.0 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --knee_stiffness 60.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 5.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0 \
    --knee_range_low 0.6 \
    --knee_range_high 0.7 \
    --fix_default_joint_pos 1


# h1_v78
python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v78 \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 1.0 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --knee_stiffness 60.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 5.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0 \
    --knee_range_low 1.0 \
    --knee_range_high 1.1 \
    --fix_default_joint_pos 1


# h1_v84
python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v84 \
    --h1_urdf_version 8 \
    --action_scale 0.5 \
    --ori_term_threshold 0.7 \
    --ankle_stiffness 60.0 \
    --hip_pitch_stiffness 60.0 \
    --knee_stiffness 60.0 \
    --ankle_damping 5.0 \
    --hip_pitch_damping 5.0 \
    --lin_vel_x_min 0.0 \
    --lin_vel_x_max 4.5 \
    --lin_vel_y_ab 0.75 \
    --ang_vel_yaw_abs 2.0 \
    --knee_range_low 0.8 \
    --knee_range_high 0.9 \
    --fix_default_joint_pos 1


# # mit robot
# python gpugym/scripts/play.py \
#     --task pbrs:humanoid \
#     --wandb_name mit_base \
#     --h1_urdf_version 8 \
#     --action_scale 0.5 \
#     --ori_term_threshold 1.0 \
#     --lin_vel_x_min 0.0 \
#     --lin_vel_x_max 4.5 \
#     --lin_vel_y_ab 0.0 \
#     --ang_vel_yaw_abs 0.0

