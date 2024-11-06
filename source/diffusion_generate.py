import torch
from model.unet import build_unet
from model.diffusion import LinearSchedule, Diffusion
import cv2
import os
from tqdm import tqdm
import numpy as np


def denormalize_image(img):
    return (img + 1) * 255 / 2


save_dir = "generated"
os.makedirs(save_dir, exist_ok=True)
with torch.no_grad():
    device = torch.device("cuda")
    denoiser = build_unet(device)
    unet_checkpoint = "model_state_UNet/latest_model.pth"
    denoiser.load_state_dict(torch.load(unet_checkpoint))
    diffuser = LinearSchedule().to(device)
    model = Diffusion(diffuser=diffuser, denoiser=denoiser)
    model.eval()
    n = 8
    image_height = 32
    image_width = 32
    input = torch.randn(n * n, 3, image_height, image_width).to(device)
    num_steps = 1000
    output = model(input, num_steps)
    output = output.squeeze(0)
    output = torch.permute(output, [0, 2, 3, 1])
    output = output.cpu().numpy()

    big_image = np.zeros((image_height * n, image_height * n, 3), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            generated_image = denormalize_image(output[i * n + j])
            big_image[i * image_height:(i + 1) * image_height, j * image_width:(j + 1) * image_width] = generated_image
    save_path = os.path.join(save_dir, f"generated_images_{n}x{n}.png")
    cv2.imwrite(save_path, big_image)
