
from crop_img import crop_img
import imageio
from tqdm import tqdm
import os

img_dir = "/mnt/hypercube/zhsha/workspace/pbrs-humanoid/imgs/h1_v84"

img_ls = sorted(os.listdir(img_dir))
img_ls = [os.path.join(img_dir, img) for img in img_ls]

turncate_num = 200
if len(img_ls) > turncate_num:
    img_ls = img_ls[:turncate_num]

images = []

for img_path in tqdm(img_ls):
    images.append(crop_img(img_path))

imageio.mimsave("test.gif", images, fps=4, loop=999)




