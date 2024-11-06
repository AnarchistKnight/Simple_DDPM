from source.model.diffusion import LinearSchedule, Diffusion
from diffusion.transformer_encoder import TransformerEncoder
# from diffusion.transformer import Transformer
import torch, argparse, os
from utils import DdpmDataset, train
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="input arguments for training DDPM")
    parser.add_argument("--model_checkpoint", type=str, default="model_state_DDPM.pth")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument("--num_epochs", type=int, default=1000000)
    parser.add_argument("--normalized_data_path", type=str, default="normalized_latent_arrays.npy")
    parser.add_argument("--min_loss_path", type=str, default="min_loss_DDPM.json")
    parser.add_argument("--num_transformer_layers", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda")

    diffuser = LinearSchedule(beta_start=0.0001, beta_end=0.01)
    denoiser = TransformerEncoder(args.num_transformer_layers, dim_model=1024, dim_ffd_hidden=1024, num_heads=8, dropout=0.1)
    # denoiser = Transformer(num_layers=args.num_transformer_layers, dim_model=1024, num_heads=8, dim_feedforward=1024,
    #                        dropout=0.1, activation=nn.ReLU(), pre_normalization=False)
    diffusion = Diffusion(diffuser, denoiser).to(device)
    if os.path.exists(args.model_checkpoint):
        diffusion.load_state_dict(torch.load(args.model_checkpoint))
        print('load model from', args.model_checkpoint, 'success')

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
    dataset = DdpmDataset(args.normalized_data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    train(model=diffusion,
          dataloader=dataloader,
          criterion=None,
          num_epochs=args.num_epochs,
          optimizer=optimizer,
          state_pth=args.model_checkpoint,
          device=device,
          min_loss_path=args.min_loss_path)