import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
class LinearSchedule(nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        # linear schedule, proposed in original ddpm paper
        self.num_steps = 1000
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, self.num_steps, dtype=torch.float32))
        self.register_buffer("sqrt_betas", torch.sqrt(self.betas))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("sqrt_alphas", torch.sqrt(self.alphas))
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("alpha_bars_prev", F.pad(self.alpha_bars[:-1], (1, 0), value=1.))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1. - self.alpha_bars))
        self.register_buffer("log_one_minus_alpha_bars", torch.log(1. - self.alpha_bars))
        self.register_buffer("sqrt_reciprocal_alpha_bars", torch.sqrt(1. / self.alpha_bars))
        self.register_buffer("sqrt_reciprocal_minus_one_alpha_bars", torch.sqrt(1. / self.alpha_bars - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("posterior_variance", self.betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coefficients_1", self.betas * torch.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars))
        self.register_buffer("posterior_mean_coefficients_2", (1. - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1. - self.alpha_bars))

        for param in self.parameters():
            param.requires_grad = False

    def sample(self, x_0, t, noise):
        return x_0
        # sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        # sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        # x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        # return x_t

    def denoise(self, x_t, noise_predict, t):
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        x_0_predict = (x_t - sqrt_one_minus_alpha_bar * noise_predict) / sqrt_alpha_bar
        return x_0_predict

    def denoise_one_step_back(self, x_t, noise_predict, t):
        beta = self.betas[t]
        sqrt_alpha = self.sqrt_alphas[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t]
        x_t_minus_one_mean = (x_t - beta * noise_predict / sqrt_one_minus_alpha_bar) / sqrt_alpha
        x_t_minus_one_variance = self.betas[t] * (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) if t > 0 else 0
        noise = torch.randn_like(x_t, device=x_t.device)
        return x_t_minus_one_mean + torch.sqrt(x_t_minus_one_variance) * noise


@torch.no_grad()
class Diffusion(nn.Module):
    def __init__(self, diffuser, denoiser):
        super().__init__()
        self.diffuser = diffuser
        self.denoiser = denoiser

    def get_losses(self, x_0):
        x_t = x_0
        noise_predict = self.denoiser(x_t)
        loss = F.mse_loss(x_0, noise_predict)
        return loss

    def forward(self, x_t, num_steps):
        x = x_t
        for t in tqdm(range(num_steps - 1, 0, -1)):
            noise_predict = self.denoiser(x)
            x = self.diffuser.denoise_one_step_back(x, noise_predict, t)
        return x

