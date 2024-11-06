import os, random, shutil
from tqdm import tqdm


src_dir = "downsample_images"
dest_dir = "downsample_images_small"
os.makedirs(dest_dir, exist_ok=True)
image_names = os.listdir(src_dir)
sampled_image_names = random.sample(image_names, 10000)
for image_name in tqdm(sampled_image_names):
    src_path = os.path.join(src_dir, image_name)
    dest_path = os.path.join(dest_dir, image_name)
    shutil.copy(src_path, dest_path)