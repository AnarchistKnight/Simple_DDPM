from utils import down_sample_image
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="downsample images")
    parser.add_argument("--source_image_dir", type=str, default="generated_images")
    parser.add_argument("--dest_image_dir", type=str, default="downsample_images")
    parser.add_argument("--dest_image_height", type=int, default=96)
    parser.add_argument("--dest_image_width", type=int, default=96)
    parser.add_argument("--skip_exists", type=bool, default=True)
    parser.add_argument("--num_samples", type=int, default=-1)
    args = parser.parse_args()

    down_sample_image(source_image_dir=args.source_image_dir,
                      dest_image_dir=args.dest_image_dir,
                      dest_image_size=(args.dest_image_height, args.dest_image_width),
                      skip_exists=args.skip_exists,
                      num_samples=args.num_samples)