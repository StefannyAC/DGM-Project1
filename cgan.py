# cgan.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# CGAN condicional por género musical compatible con data_pipeline.py
# usando WGAN-GP (critic en lugar de discriminator)
# - z: (B, z_dim), vector latente de entrada al generador.
# - cond: (B, 1) o (B,) Long (id de género en {0..3}), condición de género.
# - G(z, cond): genera secuencias tipo piano-roll (B, 128, T) en rango [0,1].
# - D(x, cond): crítico condicional que evalúa la autenticidad de una secuencia
#               dado su género, produciendo un escalar (score real/fake).
# - compute_gradient_penalty(): regularizador de Lipschitz usado en WGAN-GP.
# ============================================================
 
import torch # Para tensores
import torch.nn as nn # Para definir modelos
import torch.nn.functional as F # Para funciones de activación y pérdidas

# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------
def compute_gradient_penalty(critic_fn, real_samples, fake_samples, device):
    """
    Función para calcular el Gradient Penalty (WGAN-GP) sobre muestras interpoladas
    entre reales y generadas, forzando que ||∇_x D(x)||₂ -> 1.

    Args:
        critic_fn: Función que evalúa el Crítico D(x) y devuelve un tensor escalar por muestra (shape (B, 1) o (B,)).
        real_samples: Tensores reales x_real con shape (B, ...).
        fake_samples: Tensores generados x_fake con shape (B, ...).
        device: Dispositivo donde crear 'alpha' y realizar el cálculo.

    Returns:
        Escalar (0-D) con el valor medio del término de penalización:
            E[(||∇_x D(x_hat)||₂ - 1)²], donde x_hat = α x_real + (1-α) x_fake.
    """
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device) # (B,1,1) para broadcast en todas las dims
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)  # x_hat con grad
    critic_interpolates = critic_fn(interpolates) # D(x_hat): salida por muestra (B,1) o (B,)

    # ∇_{x_hat} D(x_hat) usando autograd:
    # - grad_outputs: tensor de unos con misma forma que la salida para propagar dL/dD = 1
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True, # conserva el grafo para seguir backward en este resultado
        retain_graph=True,
        only_inputs=True, # solo calcula grad respecto a 'interpolates'
    )[0]
    gradients = gradients.view(gradients.size(0), -1) # aplanar por muestra -> (B, num_feats)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() # E[(||g||₂ - 1)^2]

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