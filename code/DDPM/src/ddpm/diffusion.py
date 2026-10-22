"""Forward/reverse diffusion processes and the DDPM training objective."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from .config import DiffusionConfig


def linear_beta_schedule(
    timesteps: int, start: float = 1e-4, end: float = 0.02
) -> Tensor:
    return torch.linspace(start, end, timesteps, dtype=torch.float32)


def quadratic_beta_schedule(
    timesteps: int, start: float = 1e-4, end: float = 0.02
) -> Tensor:
    return torch.linspace(start**0.5, end**0.5, timesteps, dtype=torch.float32) ** 2


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    timesteps: int, start: float = 1e-4, end: float = 0.02
) -> Tensor:
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float32)
    return torch.sigmoid(betas) * (end - start) + start


def make_beta_schedule(config: DiffusionConfig) -> Tensor:
    if config.schedule == "linear":
        return linear_beta_schedule(
            config.timesteps, config.beta_start, config.beta_end
        )
    if config.schedule == "quadratic":
        return quadratic_beta_schedule(
            config.timesteps, config.beta_start, config.beta_end
        )
    if config.schedule == "cosine":
        return cosine_beta_schedule(config.timesteps)
    if config.schedule == "sigmoid":
        return sigmoid_beta_schedule(
            config.timesteps, config.beta_start, config.beta_end
        )
    raise ValueError(f"unknown beta schedule: {config.schedule}")


def extract(values: Tensor, timesteps: Tensor, x_shape: torch.Size) -> Tensor:
    """Gather per-sample values and reshape them for broadcasting over images."""

    timesteps = timesteps.to(device=values.device, dtype=torch.long)
    batch_size = timesteps.shape[0]
    out = values.gather(-1, timesteps)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """A DDPM with fixed forward variances and epsilon prediction.

    The diffusion object owns only tensors (registered as buffers), so it can be
    moved to a device independently of the U-Net.
    """

    def __init__(self, config: DiffusionConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiffusionConfig()
        self.timesteps = self.config.timesteps

        betas = make_beta_schedule(self.config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------------------------------ forward
    def q_sample(
        self, x_start: Tensor, timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        """Sample ``x_t`` directly from ``x_0`` using the closed form."""

        noise = torch.randn_like(x_start) if noise is None else noise
        sqrt_alphas_cumprod_t = extract(
            self.sqrt_alphas_cumprod, timesteps, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self, x_t: Tensor, timesteps: Tensor, noise: Tensor
    ) -> Tensor:
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    # ------------------------------------------------------------------ training
    def p_losses(
        self,
        model: nn.Module,
        x_start: Tensor,
        timesteps: Tensor,
        noise: Tensor | None = None,
        loss_type: str | None = None,
    ) -> Tensor:
        """The simplified DDPM objective ``||eps - eps_theta(x_t, t)||``."""

        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        predicted_noise = model(x_noisy, timesteps)

        loss_type = loss_type or self.config.loss_type
        if loss_type == "l1":
            return F.l1_loss(noise, predicted_noise)
        if loss_type == "l2":
            return F.mse_loss(noise, predicted_noise)
        if loss_type == "huber":
            return F.smooth_l1_loss(noise, predicted_noise)
        raise ValueError(f"unknown loss type: {loss_type}")

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def p_sample(
        self, model: nn.Module, x: Tensor, timesteps: Tensor, t_index: int
    ) -> Tensor:
        """One reverse step, following Algorithm 2 of the DDPM paper."""

        betas_t = extract(self.betas, timesteps, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, timesteps, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, timesteps) / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, timesteps, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        image_size: int,
        batch_size: int = 16,
        channels: int = 3,
        device: torch.device | str | None = None,
        progress: bool = True,
        return_all: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Full ancestral DDPM sampling (1000 network evaluations by default)."""

        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)
        shape = (batch_size, channels, image_size, image_size)
        image = torch.randn(shape, device=device)
        history: list[Tensor] = []

        iterator = tqdm(
            reversed(range(self.timesteps)),
            total=self.timesteps,
            desc="DDPM sampling",
            disable=not progress,
        )
        for index in iterator:
            timesteps = torch.full(
                (batch_size,), index, device=device, dtype=torch.long
            )
            image = self.p_sample(model, image, timesteps, index)
            if return_all:
                history.append(image.detach().cpu())

        if return_all:
            return image, history
        return image

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        image_size: int,
        batch_size: int = 16,
        channels: int = 3,
        device: torch.device | str | None = None,
        steps: int = 50,
        eta: float = 0.0,
        progress: bool = True,
        clip_denoised: bool = True,
    ) -> Tensor:
        """DDIM sampling for fast CPU inference (``eta=0`` is deterministic)."""

        if device is None:
            device = next(model.parameters()).device
        device = torch.device(device)
        steps = max(1, min(int(steps), self.timesteps))
        shape = (batch_size, channels, image_size, image_size)
        image = torch.randn(shape, device=device)

        times = (
            torch.linspace(0, self.timesteps - 1, steps, device=device)
            .round()
            .long()
            .unique()
            .sort(descending=True)
            .values
        )

        iterator = tqdm(times.tolist(), desc="DDIM sampling", disable=not progress)
        for position, current in enumerate(iterator):
            timesteps = torch.full(
                (batch_size,), int(current), device=device, dtype=torch.long
            )
            predicted_noise = model(image, timesteps)
            alpha_t = self.alphas_cumprod[timesteps].reshape(batch_size, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)

            x0 = (image - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)

            if position == len(times) - 1:
                image = x0
                break

            previous = int(times[position + 1].item())
            alpha_prev = self.alphas_cumprod[previous].reshape(1, 1, 1, 1)
            if eta > 0.0:
                sigma = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                )
            else:
                sigma = torch.zeros((), device=device, dtype=image.dtype)

            c = torch.sqrt(torch.clamp(1 - alpha_prev - sigma**2, min=0.0))
            image = torch.sqrt(alpha_prev) * x0 + c * predicted_noise
            if eta > 0.0:
                image = image + sigma * torch.randn_like(image)

        return image
