
from crop_img import crop_img
import imageio
from tqdm import tqdm
import os

img_base_dir = "/mnt/hypercube/zhsha/workspace/pbrs-humanoid/imgs"

img_dir_name_ls = os.listdir(img_base_dir)

# img_dir_name_ls = ["h1_v13"]

for img_dir_name in img_dir_name_ls:
    img_dir = os.path.join(img_base_dir, img_dir_name)
    img_ls = sorted(os.listdir(img_dir))
    img_ls = [os.path.join(img_dir, img) for img in img_ls]

    turncate_num = 200
    if len(img_ls) > turncate_num:
        img_ls = img_ls[:turncate_num]

    images = []

    for img_path in tqdm(img_ls):
        images.append(crop_img(img_path))

    gif_dir = "videos"
    # output_gif_path = os.path.join(gif_dir, f"{img_dir_name}.gif")
    output_gif_path = os.path.join(gif_dir, f"{img_dir_name}.mp4")
    # imageio.mimsave(output_gif_path, images, fps=4, loop=999)
    imageio.mimsave(output_gif_path, images, fps=4)


    print(f"save gif to {output_gif_path}")




