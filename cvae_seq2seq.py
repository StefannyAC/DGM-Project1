# cvae_seq2seq.py
# ============================================================
# CVAE condicional (género) compatible con data_pipeline.py
# usando arquitectura seq2seq con RNNs
# - X_pr: (B, 128, T)
# - events: (B, N, 3) con padding; el encoder ignora filas cero
# - cond: (B, 1) o (B,) Long (id de género en {0..3})
# ============================================================

import math
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
            nn.Linear(in_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, ev_embed), nn.ReLU(inplace=True),
        )
        self.ev_embed = ev_embed

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        """
        events: (B, N, 3) con posibles filas de ceros (padding).
                También acepta (N,3) y lo expande a (1,N,3).
        """
        if events.ndim == 2:  # (N,3) -> (1,N,3)
            events = events.unsqueeze(0)
        if events.numel() == 0:
            # sin eventos: regresa vector cero
            B = 1 if events.ndim == 2 else events.size(0)
            return torch.zeros(B, self.ev_embed, device=events.device, dtype=events.dtype)

        # asegurar float
        if not torch.is_floating_point(events):
            events = events.float()

        B, N, _ = events.shape
        if N == 0:
            return torch.zeros(B, self.ev_embed, device=events.device, dtype=events.dtype)

        # máscara: 1 si la fila tiene algún valor != 0
        mask = (events.abs().sum(dim=-1) > 0).float()         # (B, N)
        feats = self.mlp(events)                               # (B, N, ev_embed)
        summed = (feats * mask.unsqueeze(-1)).sum(dim=1)      # (B, ev_embed)
        counts = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1) # evitar /0
        return summed / counts                                 # (B, ev_embed)

# ----------------- CVAE -----------------
class CVAE(nn.Module):
    """
    forward(X, E, y) -> (X_rec, mu, logvar)
      - encoder(X, E, y) -> (mu, logvar)
      - reparameterize(mu, logvar) -> z
      - decode(z, y) -> X_rec  (B,128,T)
    """
    def __init__(
        self,
        z_dim=128,
        cond_dim=4,
        seq_len=32,
        ev_embed=64,
        cond_embed=16,
        enc_hid=256, 
        dec_hid=256
    ):
        super().__init__()
        self.z_dim = z_dim
        self.seq_len = seq_len

        self.ev_encoder  = EventEncoder(in_dim=3, ev_embed=ev_embed)
        self.cond_embed  = nn.Embedding(cond_dim, cond_embed)

        # Encoder RNN (lee (B,T,128) y resume en h_last)
        self.enc_rnn = nn.GRU(input_size=128, hidden_size=enc_hid, num_layers=1, bidirectional=True, batch_first=True)
        enc_in = 2*enc_hid + ev_embed + cond_embed
        self.fc_mu     = nn.Linear(enc_in, z_dim)
        self.fc_logvar = nn.Linear(enc_in, z_dim)

        # Decoder RNN (genera (B,T,128))
        self.h0_proj = nn.Linear(z_dim + cond_embed, dec_hid)              # estado inicial
        self.dec_rnn = nn.GRU(input_size=128 + cond_embed, hidden_size=dec_hid,
                              num_layers=1, batch_first=True)
        self.out = nn.Sequential(nn.Linear(dec_hid, 128), nn.Sigmoid())    # (B,T,128)

    # --- Encoder ---
    def encoder(self, X, E, y):
        # X: (B,128,T) -> (B,T,128)
        Xt = X.transpose(1, 2)
        h_seq, h_last = self.enc_rnn(Xt)                 # h_last: (2,B,enc_hid)
        h_last = torch.cat([h_last[-2], h_last[-1]], dim=1)  # (B,2*enc_hid)

        if y.ndim == 2 and y.size(-1) == 1: y = y.squeeze(-1)
        y_emb = self.cond_embed(y.long())                # (B,cond_embed)
        e_emb = self.ev_encoder(E)                       # (B,ev_embed)

        h = torch.cat([h_last, e_emb, y_emb], dim=1)     # (B, enc_out_dim)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        logvar = logvar.clamp(min=-10.0, max=10.0)       # estabilidad a ver si no explota
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar).clamp(min=1e-6, max=50.0)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return torch.nan_to_num(z, nan=0.0, posinf=50.0, neginf=-50.0) # evitamos NaN

    # --- Decoder ---
    def decode(self, z, y, T=None, teacher=None):
        B = z.size(0); T = T or self.seq_len
        if y.ndim == 2 and y.size(-1) == 1: 
            y = y.squeeze(-1)
        y_emb = self.cond_embed(y.long())                    # (B,cond_embed)
        h0 = torch.tanh(self.h0_proj(torch.cat([z, y_emb], dim=1))).unsqueeze(0)  # (1,B,dec_hid)

        if teacher is not None:
            # teacher: (B,128,T) -> (B,T,128+cond)
            dec_inp = torch.cat([teacher.transpose(1,2),
                                 y_emb.unsqueeze(1).repeat(1, T, 1)], dim=2)
            h_seq, _ = self.dec_rnn(dec_inp, h0)            # (B,T,dec_hid)
            Y = self.out(h_seq)                              # (B,T,128)
        else:
            # autoregresivo
            y_t = torch.zeros(B, 1, 128, device=z.device)
            cond_rep = y_emb.unsqueeze(1)                   # (B,1,cond_embed)
            h = h0; outs = []
            for _ in range(T):
                x_t = torch.cat([y_t, cond_rep], dim=2)     # (B,1,128+cond)
                h_seq, h = self.dec_rnn(x_t, h)
                y_t = self.out(h_seq)                       # (B,1,128) in [0,1]
                outs.append(y_t)
            Y = torch.cat(outs, dim=1)                      # (B,T,128)
        return Y.transpose(1, 2)                             # (B,128,T)
    
    def forward(self, X, E, y, teacher_prob: float = 1.0):
        mu, logvar = self.encoder(X, E, y)
        z = self.reparameterize(mu, logvar)
        teacher = X if self.training and teacher_prob > 0 else None
        X_rec = self.decode(z, y, T=X.size(-1), teacher=teacher)
        return X_rec, mu, logvar