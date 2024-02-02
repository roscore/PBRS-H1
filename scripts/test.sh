
export CUDA_VISIBLE_DEVICES=6

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib

# debug mode
python gpugym/scripts/train.py \
    --task pbrs:H1 \
    --h1_urdf_version 8 \
    --action_scale 1.0 \
    --headless

