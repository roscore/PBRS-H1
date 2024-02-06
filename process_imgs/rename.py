
import os
from tqdm import tqdm

tgt_img_dir = "/mnt/hypercube/zhsha/workspace/pbrs-humanoid/imgs/h1_v13"
img_file_ls = os.listdir(tgt_img_dir)

for img_name in tqdm(img_file_ls):
    idx = img_name.rstrip(".png")
    new_img_name = f"{idx.zfill(3)}.png"
    # change img file to new name
    command = f"mv {os.path.join(tgt_img_dir, img_name)} {os.path.join(tgt_img_dir, new_img_name)}"
    os.system(command)


