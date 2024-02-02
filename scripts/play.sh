
export CUDA_VISIBLE_DEVICES=0

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib
export DISPLAY=localhost:10.0

python gpugym/scripts/play.py \
    --task pbrs:H1 \
    --wandb_name h1_v19 \
    --h1_urdf_version 5 \
    --action_scale 10.0

