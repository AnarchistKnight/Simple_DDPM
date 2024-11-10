import os
from model.unet import build_unet
import torch
from utils import inference, lss
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input arguments for inferencing")
    parser.add_argument("--groundtruth_image_directory", type=str,
                        default="images/downsample_test_images")
    parser.add_argument("--pred_image_directory", type=str,
                        default="images/unet_reconstructed_downsample_test_images")
    parser.add_argument("--model_checkpoint", type=str, default="model_state_UNet.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("noise scale is", args.noise_scale)
    assert os.path.exists(args.groundtruth_image_directory)
    os.makedirs(args.pred_image_directory, exist_ok=True)

    assert torch.cuda.is_available()
    device = torch.device(args.device)
    model = build_unet(args.noise_scale, device)

    assert os.path.exists(args.model_checkpoint)
    model.load_state_dict(torch.load(args.model_checkpoint))

    inference(model=model,
              groundtruth_image_dir=args.groundtruth_image_directory,
              pred_image_dir=args.pred_image_directory,
              in_criterion=lss,
              device=device)
