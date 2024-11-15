import torch
from model.unet import build_unet
from model.diffusion import LinearSchedule, Diffusion
import cv2
import os
import numpy as np
import argparse


def denormalize_image(img):
    return (img + 1) * 255 / 2


@torch.no_grad()
def generate_image(save_dir, n, m, checkpoint, device):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    denoiser = build_unet(device)

    denoiser.load_state_dict(torch.load(checkpoint))
    diffuser = LinearSchedule().to(device)
    model = Diffusion(diffuser=diffuser, denoiser=denoiser)
    model.eval()
    image_height = 32
    image_width = 32
    input = torch.randn(n * m, 3, image_height, image_width).to(device)
    num_steps = 1000
    output = model(input, num_steps)
    output = output.squeeze(0)
    output = torch.permute(output, [0, 2, 3, 1])
    output = output.cpu().numpy()

    big_image = np.zeros((image_height * m, image_width * n, 3), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            generated_image = denormalize_image(output[i * n + j])
            big_image[i * image_height:(i + 1) * image_height, j * image_width:(j + 1) * image_width] = generated_image
    save_path = os.path.join(save_dir, f"generated_images_{n}x{m}.png")
    cv2.imwrite(save_path, big_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input arguments for diffusion generate")
    parser.add_argument("--generated_image_save_dir", type=str, default="images/generated")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest_model.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    generate_image(args.generated_image_save_dir,
                   args.n,
                   args.m,
                   args.checkpoint,
                   args.device)


