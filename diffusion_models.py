from typing import Callable, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

ModuleType = Union[str, Callable[..., nn.Module]]
SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim=100,
                 gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.KLDiv = nn.KLDivLoss(reduction='batchmean')

    def __call__(self, denoise_fn, data, common_feat=None):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (
                sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma, common_feat)

        target = y
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)
        reconstruction_errors = (D_yn - target) ** 2   
        score = torch.sqrt(torch.sum(reconstruction_errors, 1))
        return loss, score, D_yn


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2,
                             dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512):
        super().__init__()
        self.dim_t = dim_t
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t) 

        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        self.feat_proj = nn.Linear(d_in, dim_t)
        self.head = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, noise_labels, common_feat=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(
            *emb.shape)

        emb = self.time_embed(emb)
        if (common_feat is None):
            x = self.proj(x) + emb
        else:
            x = self.proj(x) + emb + self.feat_proj(common_feat)
        return self.mlp(x)

class Precond(nn.Module):
    def __init__(self,
                 denoise_fn,
                 hid_dim,
                 sigma_min=0,
                 sigma_max=float('inf'),
                 sigma_data=0.5,
                 ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, common_feat=None):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (
                sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x 
        F_x = self.denoise_fn_F(x_in.to(dtype), c_noise.flatten(), common_feat)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(self, denoise_fn, hid_dim, P_mean=-1.2, P_std=1.2,
                 sigma_data=0.5, gamma=5, opts=None, pfgmpp=False):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim,   
                               gamma=5, opts=None)

    def forward(self, x, common_feat=None):
        loss, score, reconstructed = self.loss_fn(self.denoise_fn_D, x, common_feat)
        return loss.mean(-1).mean(), score, reconstructed

def sample_step(net, num_steps, i, t_cur, t_next, x_next, common_feat=None):
    x_cur = x_next
    # Increase noise temporarily.    
    gamma = min(S_churn / num_steps, math.sqrt(
        2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() \
            * S_noise * torch.randn_like(x_cur)
    # Euler step.                  

    denoised = net(x_hat, t_hat, common_feat).to(torch.float32)
    ##################################
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur
    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised = net(x_next, t_next, common_feat).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm(net, noise, num_steps, common_feat = None):
    step_indices = torch.arange(num_steps, dtype=torch.float32,
                                device=noise.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    z = noise.to(torch.float32) * t_steps[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    with (torch.no_grad()):
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step(net, num_steps, i, t_cur, t_next, z, common_feat)
    return z


def sample_step_free(condition_net, uncondition_net, num_steps, i, t_cur, t_next, x_next, common_feat = None, lamda=None):
    x_cur = x_next
    gamma = min(S_churn / num_steps, math.sqrt(
        2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = uncondition_net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() \
            * S_noise * torch.randn_like(x_cur)


    denoised_condition = condition_net(x_hat, t_hat, common_feat=common_feat).to(torch.float32)
    denoised_uncondition = uncondition_net(x_hat, t_hat).to(torch.float32)
    d_cur_condition = (x_hat - denoised_condition) / t_hat
    d_cur_uncondition = (x_hat - denoised_uncondition) / t_hat

    ###################################
    ## Guidance Process
    ###################################
    d_cur = (1 + lamda) * d_cur_uncondition - (lamda) * d_cur_condition
    x_next = x_hat + (t_next - t_hat) * d_cur
    # Apply 2nd order correction.
    if i < num_steps - 1:
        denoised_condition = d_cur_condition(x_next, t_next, common_feat=common_feat).to(torch.float32)
        denoised_uncondition = d_cur_uncondition(x_next, t_next).to(torch.float32)
        d_prime_condition = (x_next - denoised_condition) / t_next
        d_prime_uncondition = (x_next - denoised_uncondition) / t_next
        d_prime = (1. + lamda) * d_prime_uncondition - lamda * d_prime_condition
        x_next = x_hat + (t_next - t_hat) * (
                0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm_free(condition_net, uncondition_net, noise, num_steps, common_feat = None, lamda=None):
    step_indices = torch.arange(num_steps, dtype=torch.float32,
                                device=noise.device)

    sigma_min = max(SIGMA_MIN, uncondition_net.sigma_min)
    sigma_max = min(SIGMA_MAX, uncondition_net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho #
    t_steps = torch.cat(
        [uncondition_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    z = noise.to(torch.float32) * t_steps[0]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    with (torch.no_grad()):
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step_free(condition_net, uncondition_net, num_steps, i, t_cur, t_next, z, common_feat, lamda=lamda)
    return z
