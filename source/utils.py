from torch.utils.data import Dataset

import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import cv2, os, math, json
import numpy as np
from tqdm import tqdm
import visdom


def normalize_image(image):
    return image / 255 * 2 - 1


class CustomDataset(Dataset):
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


@torch.no_grad()
def validate(model, image_dir, in_criterion, device):
    model.eval()
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
        scale = torch.rand(batch_size, 1, 1, 1, device=device)
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
    model.eval()
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
    return F.mse_loss(c_in, c_out)


def get_min_loss(path):
    with open(path, 'r') as f:
        return json.load(f)['min_loss']


def save_min_loss(path, min_loss):
    with open(path, 'w') as f:
        json.dump({'min_loss': min_loss}, f)


def train_per_epoch(model, dataloader, criterion, optimizer, device):
    batch_losses = []
    model.train()
    optimizer.zero_grad()
    for inputs in tqdm(dataloader):
        inputs = inputs.to(device)

        batch_size = inputs.shape[0]
        scale = torch.rand(batch_size, 1, 1, 1, device=device)
        noise = torch.randn_like(inputs, device=device)
        x = torch.sqrt(1 - scale) * inputs + torch.sqrt(scale) * noise
        outputs = model(x)
        loss = criterion(noise, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    epoch_loss = sum(batch_losses) / len(batch_losses)
    return epoch_loss


def train(model, dataloader, criterion, num_epochs, optimizer, model_checkpoint_dir,
          validate_every, save_every, device, validate_dataset_directory,
          min_loss_path, latest_epoch):
    min_validation_loss = get_min_loss(min_loss_path) if os.path.exists(min_loss_path) else None
    viz = visdom.Visdom()
    opts = dict(title='Diffusion Training Loss', xlabel='epoch', ylabel='Loss')
    loss_window = viz.line(X=np.array([0]), Y=np.array([0]), opts=opts)

    for epoch in range(latest_epoch, latest_epoch + num_epochs):
        epoch_loss = train_per_epoch(model, dataloader, criterion, optimizer, device)
        viz.line(X=np.array([epoch - latest_epoch]),
                 Y=np.array([epoch_loss]),
                 win=loss_window,
                 update='append',
                 name="train loss")
        print(f"train loss at {epoch}-th epoch: {epoch_loss}")

        if epoch % save_every == 0 and epoch > latest_epoch:
            save_pth = os.path.join(model_checkpoint_dir, f"{epoch}.pth")
            torch.save(model.state_dict(), save_pth)

        latest_save_path = os.path.join(model_checkpoint_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_save_path)

        if validate_every > 0 and epoch % validate_every == 0:
            validate_loss = validate(model, validate_dataset_directory, criterion, device)
            print(f"validate loss at {epoch}-th epoch: {validate_loss}")
            viz.line(X=np.array([epoch - latest_epoch]),
                     Y=np.array([validate_loss]),
                     win=loss_window,
                     update='append',
                     name="validate loss")

            if min_validation_loss is None:
                min_validation_loss = validate_loss
                save_min_loss(min_loss_path, min_validation_loss)
            elif validate_loss < min_validation_loss:
                print("model save at validation loss", validate_loss)
                min_validation_loss = validate_loss
                save_min_loss(min_loss_path, min_validation_loss)
                save_pth = os.path.join(model_checkpoint_dir, "best_validation_checkpoint.pth")
                torch.save(model.state_dict(), save_pth)
