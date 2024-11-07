import os, torch, argparse
from utils import CustomDataset, lss, train
from model.unet import build_unet
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def get_latest_model_checkpoint(model_checkpoint_dir):
    max_save_epoch = 0
    for path in os.listdir(model_checkpoint_dir):
        try:
            save_epoch = int(path.split(".")[0])
            if save_epoch > max_save_epoch:
                max_save_epoch = save_epoch
        except:
            pass
    return max_save_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments for training UNet")
    parser.add_argument("--model_checkpoint_dir", type=str, default="model_state_UNet")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=530)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--validate_dataset_directory", type=str, default="D:/playgroundv25/downsample_test_images")
    parser.add_argument("--train_dataset_directory", type=str, default="D:/playgroundv25/downsample_images")
    parser.add_argument("--min_loss_path", type=str, default="min_loss_UNet.json")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)

    dataset = CustomDataset(args.train_dataset_directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_unet(device)
    num_samples = len(dataset)
    os.makedirs(args.model_checkpoint_dir, exist_ok=True)
    args.min_loss_path = os.path.join(args.model_checkpoint_dir, args.min_loss_path)

    latest_model_checkpoint = os.path.join(args.model_checkpoint_dir, "latest_model.pth")
    if not os.path.exists(latest_model_checkpoint):
        model.init_weight()
    else:
        model.load_state_dict(torch.load(latest_model_checkpoint))
        print('load model from', latest_model_checkpoint, 'success')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    latest_epoch = get_latest_model_checkpoint(args.model_checkpoint_dir)
    train(model=model,
          dataloader=dataloader,
          criterion=lss,
          num_epochs=args.num_epochs,
          optimizer=optimizer,
          model_checkpoint_dir=args.model_checkpoint_dir,
          validate_every=args.validate_every,
          save_every=args.save_every,
          device=device,
          validate_dataset_directory=args.validate_dataset_directory,
          min_loss_path=args.min_loss_path,
          noise_scale=args.noise_scale,
          latest_epoch=latest_epoch)
