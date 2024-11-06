import os
from omegaconf import OmegaConf
from model.CAE import ConvAutoencoder
import torch
from utils import inference, lss
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input arguments for inferencing")
    parser.add_argument("--groundtruth_image_directory", type=str,
                        default="D:/playgroundv25/downsample_test_images")
    parser.add_argument("--pred_image_directory", type=str,
                        default="D:/playgroundv25/reconstructed_downsample_test_images")
    parser.add_argument("--config_file_path", type=str, default="config_CAE.yaml")
    parser.add_argument("--model_checkpoint", type=str, default="model_state.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compare_input_output", type=int, default=False)
    args = parser.parse_args()

    assert os.path.exists(args.groundtruth_image_directory)
    os.makedirs(args.pred_image_directory, exist_ok=True)

    config = OmegaConf.load(args.config_file_path)
    model = ConvAutoencoder.from_config(config)

    assert os.path.exists(args.model_checkpoint)
    model.load_state_dict(torch.load(args.model_checkpoint))

    assert torch.cuda.is_available()
    device = torch.device(args.device)
    args.compare_input_output = args.compare_input_output > 0
    inference(model=model,
              groundtruth_image_dir=args.groundtruth_image_directory,
              pred_image_dir=args.pred_image_directory,
              in_criterion=lss,
              device=device,
              compare_input_output=args.compare_input_output)
