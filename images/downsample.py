import os, random, argparse, cv2
from tqdm import tqdm


def downsample_images(source_image_dir, dest_image_dir, dest_image_size, num_samples, skip_exists=True):
    os.makedirs(dest_image_dir, exist_ok=True)
    image_names = os.listdir(source_image_dir)
    if num_samples > 0 and num_samples < len(image_names):
        image_names = random.sample(image_names, num_samples)
    for image_name in tqdm(image_names):
        source_image_path = os.path.join(source_image_dir, image_name)
        dest_image_path = os.path.join(dest_image_dir, image_name)
        if skip_exists and os.path.exists(dest_image_path):
            continue
        source_image = cv2.imread(source_image_path)
        if source_image is None:
            print("failed to load image:", source_image_path)
            continue
        dest_image = cv2.resize(source_image, dest_image_size)
        cv2.imwrite(dest_image_path, dest_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="downsample images")
    parser.add_argument("--source_image_dir", type=str, default="generated_images")
    parser.add_argument("--dest_image_dir", type=str, default="downsample_images")
    parser.add_argument("--dest_image_height", type=int, default=96)
    parser.add_argument("--dest_image_width", type=int, default=96)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--skip_exists", type=bool, default=True)
    args = parser.parse_args()

    downsample_images(source_image_dir=args.source_image_dir,
                      dest_image_dir=args.dest_image_dir,
                      dest_image_size=(args.dest_image_height, args.dest_image_width),
                      num_samples=args.num_samples,
                      skip_exists=args.skip_exists)