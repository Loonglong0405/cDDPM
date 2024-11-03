import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from functools import reduce
# Diffusion parameters
T = 400
noise_schedule = np.linspace(1e-4, 0.05, T).tolist()  # beta_t
alpha_t = [1 - beta_t for beta_t in noise_schedule]
alpha_bar_t = [alpha_t[0] if s == 0 else reduce((lambda x, y: x * y), alpha_t[:s]) for s in range(len(noise_schedule))]
beta_tild_t = [noise_schedule[0] if t == 0 else ((1 - alpha_bar_t[t - 1]) / (1 - alpha_bar_t[t])) * noise_schedule[t]
               for t in range(len(noise_schedule))]

diff_params = [noise_schedule, alpha_t, alpha_bar_t, beta_tild_t]

alpha_bar_t = torch.tensor(alpha_bar_t)
class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # n[1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context):
        h = self.num_heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(-1, h, t.shape[1], t.shape[2] // h), (q, k, v))
        attention = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attention = F.softmax(attention, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attention, v)
        out = out.reshape(x.shape)
        return self.to_out(out)

class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        hidden_dim = 256  # 添加一个隐藏维度

        self.linear1 = nn.Linear(cond_length, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # 添加注意力层
        self.attention = CrossAttention(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 输出层
        self.linear2 = nn.Linear(hidden_dim, target_dim)

    def forward(self, x):
        # 初始特征提取
        x = self.linear1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x, 0.4)

        # 自注意力处理
        identity = x
        x = self.attention(x, x)  # 自注意力
        x = self.norm2(x + identity)  # 添加残差连接

        # 输出投影
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step_embed, conditioner):
        diffusion_step = self.diffusion_projection(diffusion_step_embed).unsqueeze(-1)
        y = x + diffusion_step

        conditioner = self.conditioner_projection(conditioner)
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / sqrt(2.0), skip


class EpsilonThetaCond(nn.Module):
    def __init__(
            self,
            target_dim=24,#这里的参数没用到,实际上是具体计算了特征的维度
            cond_length=600,
            time_emb_dim=32,
            residual_layers=8,
            residual_channels=8,
            dilation_cycle_length=2,
            residual_hidden=64,
    ):
        super().__init__()
        self.cond_length = cond_length
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1
        )

        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden,
            max_steps=len(noise_schedule)
        )

        self.cond_sampler = CondUpsampler(self.cond_length, target_dim)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 1, 1)

    def forward(self, input, time, condition=None):
        if torch.is_tensor(time):
            time = time.item()

        diffusion_step = self.diffusion_embedding(time)
        conditioner = self.cond_sampler(condition)
        x = self.input_projection(input)
        x = F.relu(x)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, conditioner)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

    def sample(self, n_s=1, x_cond=None):
        """
        :param n_s: number of scenarios
        :param x_cond: context (weather forecasts, etc) into an array of shape (self.cond_in,)
        :return: samples into an array of shape (nb_samples, self.in_size)
        """
        # Generate samples from Gaussian noise
        x_t = torch.randn(n_s, 144)
        if x_cond is None:
            context = None
        else:
            context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, self.cond_length)).float()

        for t in range(T - 1, -1, -1):
            z = torch.randn(x_t.shape) if t > 1 else 0
            t = torch.tensor(t)
            t = t.to(device)
            if context is not None:
                eps_theta_t = self.forward(x_t.unsqueeze(1).to(device), t, context.unsqueeze(1).to(device))
            else:
                eps_theta_t = self.forward(x_t.unsqueeze(1).to(device), t)

            eps_theta_t = eps_theta_t.squeeze(1)
            mu = (x_t - ((noise_schedule[t] / sqrt(1 - alpha_bar_t[t])) * eps_theta_t.to('cpu'))) / sqrt(alpha_t[t])

            sigma = torch.ones(mu.shape) * sqrt(beta_tild_t[t])

            # normal = Normal(torch.tensor(mu), torch.tensor(sigma)) # One way of sampling
            # x_t = normal.sample()
            x_t = mu + sigma * z  # other way of sampling

        scenarios = x_t

        return scenarios