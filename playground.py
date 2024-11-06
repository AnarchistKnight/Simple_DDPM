import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm

source_dir = "D:/playgroundv25/downsample_images"
dest_dir = "D:/playgroundv25/downsample_images_small"
os.makedirs(dest_dir, exist_ok=True)
names = os.listdir(source_dir)[:30000]
for name in tqdm(names):
    source_path = os.path.join(source_dir, name)
    dest_path = os.path.join(dest_dir, name)
    shutil.copy(source_path, dest_path)
