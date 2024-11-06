import os, torch, argparse
from utils import CaeDataset, lss, train
from model.CAE import ConvAutoencoder
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments for training convolutional autoencoder")
    parser.add_argument("--config_file_path", type=str, default="config.yaml")
    parser.add_argument("--model_checkpoint", type=str, default="model_state_CAE.pth")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=530)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--dataset_directory", type=str, default="D:/playgroundv25/downsample_images")
    parser.add_argument("--min_loss_path", type=str, default="min_loss_CAE.json")
    args = parser.parse_args()
    
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)

    config = OmegaConf.load(args.config_file_path)
    model = ConvAutoencoder.from_config(config)
    if os.path.exists(args.model_checkpoint):
        model.load_state_dict(torch.load(args.model_checkpoint))
        print('load model from', args.model_checkpoint, 'success')
    else:
        model.init_weight()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = CaeDataset(args.dataset_directory)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train(model=model,
          dataloader=dataloader,
          criterion=lss,
          num_epochs=args.num_epochs,
          optimizer=optimizer,
          state_pth=args.model_checkpoint,
          device=device,
          min_loss_path=args.min_loss_path)
