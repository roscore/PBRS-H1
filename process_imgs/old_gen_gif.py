

import imageio
import os
from tqdm import tqdm

output_dir = "video_gifs"

png_dir = "video_frames/gen_2ymax"

print("-" * 50)
print("png_dir: ", png_dir)
print("-" * 50)

termination = 260

pngs = os.listdir(png_dir)
pngs.sort(key=lambda x: int(x.split('.')[0]))

if termination > 0:
    pngs = pngs[:termination]

# get basename of png dir
base_name = os.path.basename(png_dir)
output_path = os.path.join(output_dir, base_name + ".gif")

filenames = [os.path.join(png_dir, f) for f in pngs]

images = []
for filename in tqdm(filenames):
    images.append(imageio.imread(filename))
imageio.mimsave(output_path, images, fps=4, loop=999)



