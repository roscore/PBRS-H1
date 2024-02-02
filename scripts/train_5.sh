
export CUDA_VISIBLE_DEVICES=5

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib

python gpugym/scripts/train.py \
    --task pbrs:H1 \
    --headless \
    --wandb_project "pbrs3" \
    --wandb_name "h1_v18" \
    --h1_urdf_version 5 \
    --action_scale 5.0

# # debug mode
# python gpugym/scripts/train.py \
#     --task pbrs:H1 \
#     --h1_urdf_version 8 \
#     --headless

