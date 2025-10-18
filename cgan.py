# cgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
def compute_gradient_penalty(critic_fn, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic_fn(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# ------------------------------------------------------------
# Generator: G(z, cond)
# ------------------------------------------------------------
class Generator(nn.Module):
    """
    Generador condicional: produce secuencias tipo piano-roll
    a partir de un vector latente z y una condición (género).
    """
    def __init__(self, z_dim=128, cond_dim=4, seq_len=128, output_dim=128, hidden_dim=512):
        """
        Args:
            z_dim: Dimensión del vector latente.
            cond_dim: Número de clases de condición (géneros).
            seq_len: Longitud temporal de salida.
            output_dim: Número de notas (normalmente 128).
            hidden_dim: Dimensión interna de capas FC.
        """
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Embedding para la condición (género)
        self.embed = nn.Embedding(cond_dim, cond_dim)

        # Red completamente conectada
        self.net = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, output_dim * seq_len),
            nn.Sigmoid()  # salida en rango [0,1]
        )

    def forward(self, z, cond):
        """
        Args:
            z: Tensor (B, z_dim)
            cond: Tensor (B, 1) con IDs de género
        Returns:
            piano_roll: Tensor (B, output_dim, seq_len)
        """
        cond_emb = self.embed(cond.squeeze(1))         # (B, cond_dim)
        x = torch.cat([z, cond_emb], dim=1)           # (B, z_dim + cond_dim)
        out = self.net(x)
        return out.view(-1, self.output_dim, self.seq_len)


# ------------------------------------------------------------
# Critic / Discriminator: D(x, cond)
# ------------------------------------------------------------
class Critic(nn.Module):
    """
    Crítico condicional (WGAN-GP) que evalúa la autenticidad
    de una secuencia dada su etiqueta de género.
    """
    def __init__(self, cond_dim=4, seq_len=128, input_dim=128, hidden_dim=512):
        """
        Args:
            cond_dim: Número de clases de condición.
            seq_len: Longitud temporal de entrada.
            input_dim: Número de notas (128).
            hidden_dim: Dimensión interna.
        """
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed = nn.Embedding(cond_dim, cond_dim)

        # Capa inicial convolucional para aprender patrones temporales
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim + cond_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Proyección final
        conv_out_len = seq_len // 4  # por los strides
        self.fc = nn.Sequential(
            nn.Linear((hidden_dim // 2) * conv_out_len, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1)  # salida escalar
        )

    def forward(self, x, cond):
        """
        Args:
            x: Tensor (B, input_dim, seq_len)
            cond: Tensor (B, 1)
        Returns:
            Real/Fake score: (B, 1)
        """
        cond_emb = self.embed(cond.squeeze(1))                # (B, cond_dim)
        cond_map = cond_emb.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, cond_dim, T)
        x_cond = torch.cat([x, cond_map], dim=1)             # (B, input_dim+cond_dim, T)
        features = self.conv(x_cond)
        features = features.view(features.size(0), -1)
        return self.fc(features)
