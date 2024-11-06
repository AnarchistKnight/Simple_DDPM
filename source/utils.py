from torch.utils.data import Dataset
from diffusers import DiffusionPipeline
from diffusers import EDMDPMSolverMultistepScheduler
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2, os, math, json
import numpy as np
from tqdm import tqdm
# from torch.cuda.amp import autocast
import visdom
import random


def normalize_image(image):
    return image / 255 * 2 - 1


class CaeDataset(Dataset):
    def __init__(self, input_dir):
        assert os.path.exists(input_dir)
        input_transform = transforms.ToTensor()
        self.input_images = []
        print(f'there are {len(os.listdir(input_dir))} images in the train set.')
        for input_image_name in tqdm(os.listdir(input_dir)):
            if input_image_name.split('.')[-1] not in ['jpg', 'png']:
                continue
            input_image_path = os.path.join(input_dir, input_image_name)
            input_image = cv2.imread(input_image_path)
            input_image = np.array(normalize_image(input_image), dtype=np.float32)
            input_image = input_transform(input_image)
            self.input_images.append(input_image)
        self.length = len(self.input_images)
        print('complete loading')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.input_images[idx]
        # input_image = np.array(input_image / 255, dtype=np.float32)
        # input_image = np.array(input_image / 255.0, dtype=np.float32)
        # return self.input_transform(input_image)


class DdpmDataset(Dataset):
    def __init__(self, npz_path):
        self.array = np.load(npz_path)[:150, :]

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, index):
        return self.array[index]


def down_sample_image(source_image_dir, dest_image_dir, dest_image_size, skip_exists, num_samples):
    os.makedirs(dest_image_dir, exist_ok=True)
    image_names = os.listdir(source_image_dir)
    if num_samples > 0:
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


def generate_image_by_prompt(num_samples,
                             dest_image_dir,
                             prompt="anime-style avatar, totally white background",
                             model_name="playgroundai/playground-v2.5-1024px-aesthetic",
                             torch_dtype=torch.float16,
                             variant="fp16",
                             num_inference_steps=50,
                             guidance_scale=0.05,
                             device="cuda"):
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype, variant=variant).to(device)
    # # Optional: Use DPM++ 2M Karras scheduler for crisper fine details
    pipe.scheduler = EDMDPMSolverMultistepScheduler()
    # create a for loop to iterate through the prompts array
    os.makedirs(dest_image_dir, exist_ok=True)
    start_index = len(os.listdir(dest_image_dir))
    for index in range(start_index, start_index + num_samples):
        image = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        image_save_path = os.path.join(dest_image_dir, f"{index}.png")
        image.save(image_save_path)


@torch.no_grad()
def validate(model, image_dir, in_criterion, device, noise_scale):
    model.eval()  # 将模型设置为评估模式
    model = model.to(device)
    input_transform = transforms.ToTensor()
    losses = []
    for image_name in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_name)
        in_img = cv2.imread(image_path)
        img = np.array(normalize_image(in_img), dtype=np.float32)
        img = input_transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        batch_size = 1
        scale = noise_scale * torch.rand(batch_size, 1, 1, 1, device=device)
        noise = torch.randn_like(img, device=device)
        x = torch.sqrt(1 - scale) * img + torch.sqrt(scale) * noise
        outputs = model(x)
        loss = in_criterion(noise, outputs).item()
        losses.append(loss)
    return sum(losses) / len(losses)


@torch.no_grad()
def inference(model, groundtruth_image_dir, pred_image_dir, in_criterion, device, compare_input_output=False):
    assert os.path.exists(groundtruth_image_dir)
    os.makedirs(pred_image_dir, exist_ok=True)
    model.eval()  # 将模型设置为评估模式
    model = model.to(device)
    input_transform = transforms.ToTensor()
    losses = []
    PSNRs = []
    for image_name in tqdm(os.listdir(groundtruth_image_dir)):
        image_path = os.path.join(groundtruth_image_dir, image_name)
        os.makedirs(pred_image_dir, exist_ok=True)
        in_img = cv2.imread(image_path)
        img = np.array(in_img / 255, dtype=np.float32)
        img = input_transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)
        scale = model.noise_scale * torch.rand(1, device=device)
        noise = torch.randn_like(img, device=device)
        outputs = model(img, scale.view(-1, 1, 1, 1), noise)
        loss = in_criterion(noise, outputs).item()
        # out_img = model(img)
        # loss = in_criterion(img, out_img).item()
        PSNR = 10 * math.log(1 / loss, 10)
        PSNRs.append(PSNR)
        losses.append(loss)
        out_img = out_img.squeeze(0)
        out_img = 255 * out_img
        out_img = out_img.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
        if compare_input_output:
            out_img = cv2.hconcat([in_img, out_img])
        out_image_path = os.path.join(pred_image_dir, image_name)
        # print(f"psnr of image {out_image_path} is {PSNR}")
        cv2.imwrite(out_image_path, out_img)
        # print(f'inference loss of image {out_image_path} is {loss.item()}')
    print("average loss is", sum(losses) / len(losses))
    print("average PSNR is", sum(PSNRs) / len(PSNRs))


def lss(c_in, c_out):
    # before permutation: batch_size x num_channels x height x width
    # c_in = torch.permute(c_in, dims=[0, 2, 3, 1])
    # c_out = torch.permute(c_out, dims=[0, 2, 3, 1])
    return F.mse_loss(c_in, c_out)


def weighted_mse_lss(c_in, c_out, scale):
    loss = F.mse_loss(c_in, c_out, reduction="none")
    loss = scale * loss
    return loss.mean()


def get_min_loss(path):
    with open(path, 'r') as f:
        return json.load(f)['min_loss']


def save_min_loss(path, min_loss):
    with open(path, 'w') as f:
        json.dump({'min_loss': min_loss}, f)


def train_per_epoch(model, dataloader, criterion, optimizer, device, noise_scale, loss_window,
                    print_loss_every, viz, epoch, losses):
    batch_losses = []
    model.train()
    num_batches = len(dataloader)
    optimizer.zero_grad()
    count = 0
    for inputs in tqdm(dataloader):
        inputs = inputs.to(device)

        batch_size = inputs.shape[0]
        scale = noise_scale * torch.rand(batch_size, 1, 1, 1, device=device)
        noise = torch.randn_like(inputs, device=device)
        x = torch.sqrt(1 - scale) * inputs + torch.sqrt(scale) * noise
        outputs = model(x)
        loss = criterion(noise, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss.backward()
        # if count > 0 and count % 4 == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()

        losses.append(loss.item())
        if len(losses) > print_loss_every:
            losses.pop(0)
        batch_losses.append(loss.item())

        # viz.line(X=np.array([epoch * num_batches + batch_index]), Y=np.array([sum(losses) / len(losses)]),
        #          win=loss_window, update='append', name="average batch loss")
        # viz.line(X=np.array([epoch * num_batches + batch_index]), Y=np.array([loss.item()]),
        #          win=loss_window, update='append', name="batch loss")
    epoch_loss = sum(batch_losses) / len(batch_losses)
    viz.line(X=np.array([epoch]), Y=np.array([epoch_loss]), win=loss_window, update='append', name="epoch loss")
    return epoch_loss


def train(model, dataloader, criterion, num_epochs, optimizer, model_checkpoint_dir,
          validate_every, save_every, device, validate_dataset_directory,
          min_loss_path, noise_scale, latest_epoch):
    min_validation_loss = get_min_loss(min_loss_path) if os.path.exists(min_loss_path) else None
    viz = visdom.Visdom()
    opts = dict(title='Diffusion Training Loss', xlabel='epoch', ylabel='Loss')
    loss_window = viz.line(X=np.array([0]), Y=np.array([0]), opts=opts)

    losses = []
    for epoch in range(latest_epoch, latest_epoch + num_epochs):
        epoch_loss = train_per_epoch(model, dataloader, criterion, optimizer, device, noise_scale, loss_window,
                                     100, viz, epoch - latest_epoch, losses)
        # viz.line(X=np.array([epoch]), Y=np.array([epoch_loss]), win=loss_window, update='append', name="train loss")
        print(f"train loss at {epoch}-th epoch: {epoch_loss}")

        if epoch % save_every == 0 and epoch > latest_epoch:
            save_pth = os.path.join(model_checkpoint_dir, f"{epoch}.pth")
            torch.save(model.state_dict(), save_pth)

        latest_save_path = os.path.join(model_checkpoint_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_save_path)

        if validate_every > 0 and epoch > latest_epoch and epoch % validate_every == 0:
            validate_loss = validate(model, validate_dataset_directory, criterion, device, noise_scale)
            print(f"validate loss at {epoch}-th epoch: {validate_loss}")
            viz.line(X=np.array([epoch - latest_epoch]), Y=np.array([validate_loss]), win=loss_window,
                     update='append', name="validate loss")

            if min_validation_loss is None:
                min_validation_loss = validate_loss
                save_min_loss(min_loss_path, min_validation_loss)
            elif validate_loss < min_validation_loss:
                print("model save at validation loss", validate_loss)
                min_validation_loss = validate_loss
                save_min_loss(min_loss_path, min_validation_loss)
                save_pth = os.path.join(model_checkpoint_dir, "best_validation_checkpoint.pth")
                torch.save(model.state_dict(), save_pth)
