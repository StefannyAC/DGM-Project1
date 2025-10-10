# cvae.py
# ============================================================
# CVAE condicional (género) compatible con data_pipeline.py
# - X_pr: (B, 128, T)
# - events: (B, N, 3) con padding; el encoder ignora filas cero
# - cond: (B, 1) Long  (id de género)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ Event Encoder ------------
class EventEncoder(nn.Module):
    """
    (B, N, 3) -> (B, ev_embed). Ignora filas de padding (todo-cero).
    """
    def __init__(self, in_dim=3, ev_embed=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, ev_embed), nn.ReLU(inplace=True),
        )
        self.ev_embed = ev_embed

    def forward(self, events):                   # events: (B, N, 3)
        if events.ndim == 2:                     # caso borde: (N,3) sin batch
            events = events.unsqueeze(0)
        B, N, _ = events.shape
        if N == 0:
            return events.new_zeros(B, self.ev_embed)

        # máscara: 1 si la fila tiene algún valor != 0
        mask = (events.abs().sum(dim=-1) > 0).float()         # (B, N)
        feats = self.mlp(events)                               # (B, N, ev_embed)
        summed = (feats * mask.unsqueeze(-1)).sum(dim=1)      # (B, ev_embed)
        counts = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1) # evitar /0
        return summed / counts                                 # (B, ev_embed)


# ------------ Piano-roll Encoder ------------
class PRollEncoder(nn.Module):
    """
    Conv1d sobre (B, 128, T) -> (B, pr_embed)
    """
    def __init__(self, pr_embed=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(256, pr_embed, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
        )
        self.pr_embed = pr_embed

    def forward(self, x):                        # (B, 128, T)
        h = self.conv(x)                         # (B, pr_embed, T//4)
        return h.mean(dim=-1)                    # (B, pr_embed)


# ----------------- CVAE -----------------
class CVAE(nn.Module):
    """
    forward(X, E, y) -> (X_rec, mu, logvar)
    - encoder(X, E, y) -> (mu, logvar)
    - reparameterize(mu, logvar) -> z
    - decode(z, y) -> X_rec
    """
    def __init__(
        self,
        z_dim=128,
        cond_dim=4,
        seq_len=128,
        pr_embed=256,
        ev_embed=64,
        cond_embed=16,
        dec_hidden=256,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.seq_len = seq_len

        self.pr_encoder  = PRollEncoder(pr_embed=pr_embed)
        self.ev_encoder  = EventEncoder(in_dim=3, ev_embed=ev_embed)
        self.cond_embed  = nn.Embedding(cond_dim, cond_embed)

        enc_in = pr_embed + ev_embed + cond_embed
        self.fc_mu     = nn.Linear(enc_in, z_dim)
        self.fc_logvar = nn.Linear(enc_in, z_dim)

        # Decoder (de (z + emb_cond) -> (B, 128, T))
        t_red = seq_len // 4
        self.fc_dec = nn.Sequential(
            nn.Linear(z_dim + cond_embed, dec_hidden * t_red),
            nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(dec_hidden, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # salida en [0,1]
        )

    # ---- Latent path ----
    def encoder(self, X, E, y):
        """
        X: (B, 128, T), E: (B, N, 3), y: (B, 1) Long
        """
        pr_feat = self.pr_encoder(X)                     # (B, pr_embed)
        ev_feat = self.ev_encoder(E)                     # (B, ev_embed)
        y_emb   = self.cond_embed(y.squeeze(-1).long())  # (B, cond_embed)
        h = torch.cat([pr_feat, ev_feat, y_emb], dim=1)  # (B, enc_in)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- Decoder ----
    def decode(self, z, y):
        y_emb = self.cond_embed(y.squeeze(-1).long())    # (B, cond_embed)
        d_in  = torch.cat([z, y_emb], dim=1)             # (B, z+cond)
        x = self.fc_dec(d_in)                             # (B, dec_hidden * T//4)
        B = x.size(0); t_red = self.seq_len // 4
        x = x.view(B, -1, t_red)                          # (B, dec_hidden, T//4)
        return self.deconv(x)                             # (B, 128, T)

    def forward(self, X, E, y):
        mu, logvar = self.encoder(X, E, y)
        z = self.reparameterize(mu, logvar)
        X_rec = self.decode(z, y)
        return X_rec, mu, logvar
