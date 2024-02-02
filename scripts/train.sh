
export CUDA_VISIBLE_DEVICES=4

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib


python gpugym/scripts/train.py \
    --task=pbrs:H1 \
    --headless \
    --wandb_project "pbrs3" \
    --wandb_name "HIP_YAW_0.78_0.78_SOFT_POS_L_1.0_urdf_v7"

# # debug mode
# python gpugym/scripts/train.py \
#     --task=pbrs:H1 \
#     --headless

