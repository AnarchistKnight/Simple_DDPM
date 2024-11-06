import argparse, torch, os, cv2
from tqdm import tqdm
import numpy as np
from omegaconf import OmegaConf
from model.CAE import ConvAutoencoder
import torchvision.transforms as transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for normalizing the latent encoding of the input image")
    parser.add_argument("--config_file_path", type=str, default="config.yaml")
    parser.add_argument("--dataset_directory", type=str, default="downsample_images")
    parser.add_argument("--model_checkpoint", type=str, default="model_state_CAE.pth")
    parser.add_argument("--latent_arrays", type=str, default="latent_arrays.npy")
    parser.add_argument("--normalized_latent_arrays", type=str, default="normalized_latent_arrays.npy")
    parser.add_argument("--latent_mean", type=str, default="latent_means.npy")
    parser.add_argument("--latent_standard_deviation", type=str, default="latent_standard_deviations.npy")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = OmegaConf.load(args.config_file_path)
    model = ConvAutoencoder.from_config(config)
    assert os.path.exists(args.model_checkpoint)
    model.load_state_dict(torch.load(args.model_checkpoint))
    print('load model from', args.model_checkpoint, 'success')
    model.to(device).eval()
    input_transform = transforms.ToTensor()

    with torch.no_grad():
        latent_array_list = []
        for image_name in tqdm(os.listdir(args.dataset_directory)):
            image_path = os.path.join(args.dataset_directory, image_name)
            image = cv2.imread(image_path)
            in_array = np.array(image / 255, dtype=np.float32)
            in_tensor = input_transform(in_array)
            in_tensor = in_tensor.unsqueeze(0)
            in_tensor = in_tensor.to(device)
            latent_tensor = model.encoder(in_tensor).view(-1)
            latent_array = latent_tensor.cpu().numpy()
            latent_array_list.append(latent_array)

        latent_arrays = np.array(latent_array_list)
        np.save(args.latent_arrays, latent_arrays)

        mean = np.mean(latent_arrays, axis=0)
        np.save(args.latent_mean, mean)

        standard_deviation = np.std(latent_arrays, axis=0)
        np.save(args.latent_standard_deviation, standard_deviation)

        normalized_latent_array_list = []
        for latent_array in latent_array_list:
            normalized_latent_array = (latent_array - mean) / standard_deviation
            normalized_latent_array_list.append(normalized_latent_array)
        normalized_latent_arrays = np.array(normalized_latent_array_list)
        np.save(args.normalized_latent_arrays, normalized_latent_arrays)

