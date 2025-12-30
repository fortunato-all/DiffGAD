import torch
import torch.nn.functional as F
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())  # 
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input/t)
    sum = torch.sum(ex, axis=axis)
    return ex/sum

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_noises(timesteps=500):
    # calculations for noises
    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod