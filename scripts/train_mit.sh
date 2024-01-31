
export CUDA_VISIBLE_DEVICES=1

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib

# python gpugym/scripts/train.py --task=pbrs:humanoid --headless

python gpugym/scripts/train.py \
    --task=pbrs:H1 \
    --headless \
    --wandb_name "test"

