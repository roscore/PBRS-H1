
export CUDA_VISIBLE_DEVICES=1

export LD_LIBRARY_PATH=/home/zhsha/miniconda3/envs/pbrs/lib

# # H1 debug mode
# python gpugym/scripts/train.py \
#     --task pbrs:H1 \
#     --h1_urdf_version 8 \
#     --action_scale 1.0 \
#     --headless


# MIT debug mode
python gpugym/scripts/train.py \
    --headless \
    --task pbrs:humanoid 

    

