# train_hybrid.py

import math
import logging
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_pipeline import build_loader
from cvae import CVAE
from cgan import Generator, Critic, compute_gradient_penalty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def as_long_labels(y):
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y.long()

def reconfigure_cvae_for_T(cvae: CVAE, new_T: int, device: torch.device):
    """Ajusta el decoder del CVAE al nuevo seq_len (T)."""
    if cvae.seq_len == new_T:
        return
    cvae.seq_len = new_T
    cvae.t_reduced = math.ceil(new_T / 4)
    dec_hidden = cvae.deconv[0].in_channels  # 256 en tu CVAE
    cvae.fc_dec = torch.nn.Sequential(
        torch.nn.Linear(cvae.z_dim + cvae.cond_embed.embedding_dim, dec_hidden * cvae.t_reduced),
        torch.nn.ReLU(inplace=True),
    ).to(device)

def rebuild_gd_for_T(z_dim: int, cond_dim: int, new_T: int, device: torch.device):
    """Crea nuevas instancias de G y D para un seq_len distinto."""
    G_new = Generator(z_dim=z_dim, cond_dim=cond_dim, seq_len=new_T).to(device)
    D_new = Critic(cond_dim=cond_dim, seq_len=new_T).to(device)
    return G_new, D_new

def transfer_matching_weights(src_model: torch.nn.Module, dst_model: torch.nn.Module):
    """Copia pesos compatibles (misma clave y shape) de src a dst."""
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()
    keep = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}
    dst_sd.update(keep)
    dst_model.load_state_dict(dst_sd, strict=False)

def check_shapes_once(X, E, y, cvae, G, D, T, z_dim, cond_dim):
    B, C, TT = X.shape
    assert C == 128, f"X debe ser (B,128,T); got (B,{C},{TT})"
    assert TT == T,  f"T del batch ({TT}) != seq_len esperado ({T})"
    y_flat = as_long_labels(y)
    assert y_flat.shape[0] == B, "Batch size en y no coincide con X"
    assert y_flat.min() >= 0 and y_flat.max() < cond_dim, f"y fuera de rango [0,{cond_dim-1}]"

    with torch.no_grad():
        mu, logvar = cvae.encoder(X, E, y)
        assert mu.shape == (B, z_dim) and logvar.shape == (B, z_dim), "Encoder shapes inválidos"
        z = cvae.reparameterize(mu, logvar)
        X_rec = cvae.decode(z, y)
        assert X_rec.shape == (B, 128, T), f"Decoder shape inválido: {tuple(X_rec.shape)}!=(B,128,{T})"
        X_fake = G(z, y)
        r_out = D(X, y)
        f_out = D(X_fake, y)
        assert r_out.shape[0] == B and f_out.shape[0] == B, "Critic batch mismatch"

def train_hybrid_epoch(cvae, G, D, dataloader, opts, device, cfg, class_weights):
    cvae.train(); G.train(); D.train()
    opt_cvae, opt_g, opt_d = opts

    for batch in tqdm(dataloader, desc="Entrenando híbrido CVAE + C-GAN"):
        X = batch["piano_roll"].to(device).float().contiguous()
        E = batch["events"].to(device).float()
        y = batch["conditions"].to(device)
        y_flat = as_long_labels(y)

        # Evitar batch=1 (por cgan.py 'squeeze()' sin eje)
        if X.size(0) == 1:
            continue

        # ----- Critic (WGAN-GP) -----
        for _ in range(cfg["critic_iters"]):
            opt_d.zero_grad()

            with torch.no_grad():
                mu, logvar = cvae.encoder(X, E, y)
                z = cvae.reparameterize(mu, logvar)
                X_fake = G(z, y).detach()

            real_out = D(X, y).squeeze(1)
            fake_out = D(X_fake, y).squeeze(1)

            gp = compute_gradient_penalty(lambda t: D(t, y), X.detach(), X_fake.detach(), device)
            loss_d = -torch.mean(real_out) + torch.mean(fake_out) + cfg["lambda_gp"] * gp
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_d.step()

        # ----- Generator + CVAE -----
        opt_g.zero_grad()
        opt_cvae.zero_grad()

        # ELBO (MSE) con ponderación por clase
        X_rec, mu, logvar = cvae(X, E, y)
        recon = F.mse_loss(X_rec, X, reduction='none').sum(dim=(1,2))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

        w = class_weights[y_flat]
        elbo = (w * (recon + cfg["beta"] * kl)).mean()

        with torch.no_grad():
            z = cvae.reparameterize(mu, logvar)
        X_fake = G(z, y)
        g_out = D(X_fake, y).squeeze(1)
        loss_g = -torch.mean(g_out)

        loss = cfg["alpha"] * elbo + cfg["gamma"] * loss_g
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt_g.step()
        opt_cvae.step()

def main():
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nUsing Device: {device}\n")

    # --- config ---
    midi_root = "dataset/data/Lakh_MIDI_Dataset_Clean"
    csv_path  = "dataset/data/lakh_clean_merged_homologado.csv"

    curriculum = [32, 64, 128]   # seq_len reales por etapa
    epochs_per = [2, 2, 2]       # ajusta a gusto

    z_dim = 128
    cond_dim = 4
    pr_embed = 256
    ev_embed = 64
    cond_embed = 16

    base_batch_size = 64   # ← batch fijo
    num_workers = 0

    # modelos (arrancan en la primera T del currículo)
    cvae = CVAE(
        z_dim=z_dim, cond_dim=cond_dim, seq_len=curriculum[0],
        pr_embed=pr_embed, ev_embed=ev_embed, cond_embed=cond_embed
    ).to(device)
    G = Generator(z_dim=z_dim, cond_dim=cond_dim, seq_len=curriculum[0]).to(device)
    D = Critic(cond_dim=cond_dim, seq_len=curriculum[0]).to(device)

    # Carga preentrenos (si existen)
    ckpt_cvae = Path("checkpoints/cvae_pretrained.pth")
    if ckpt_cvae.exists():
        cvae.load_state_dict(torch.load(ckpt_cvae, map_location=device))
        logging.info("CVAE preentrenada cargada.")
    ckpt_g = Path("checkpoints/generator_pretrained.pth")
    if ckpt_g.exists():
        G.load_state_dict(torch.load(ckpt_g, map_location=device))
        logging.info("Generator preentrenado cargado.")
    ckpt_d = Path("checkpoints/critic_pretrained.pth")
    if ckpt_d.exists():
        D.load_state_dict(torch.load(ckpt_d, map_location=device))
        logging.info("Critic preentrenado cargado.")

    # Opts iniciales
    opt_cvae = torch.optim.Adam(cvae.parameters(), lr=5e-5)
    opt_g    = torch.optim.Adam(G.parameters(),    lr=1e-4, betas=(0.5, 0.999))
    opt_d    = torch.optim.Adam(D.parameters(),    lr=1e-4, betas=(0.5, 0.999))

    cfg = {
        "beta": 1.0,
        "alpha": 1.0,
        "gamma": 0.5,
        "critic_iters": 5,
        "lambda_gp": 10.0,
        "recon_loss": "mse",
    }

    # === Curriculum real: 32 -> 64 -> 128 ===
    # T representa la longitud temporal (seq_len)
    # En cada etapa se reconfiguran CVAE, G y D para el nuevo T
    prev_G, prev_D = G, D  # para transferir pesos entre etapas
    for stage, (T, n_epochs) in enumerate(zip(curriculum, epochs_per), 1):
        logging.info(f"\n===== CURRICULUM STAGE {stage}: seq_len={T}, epochs={n_epochs} =====")

        # DataLoader para el nuevo T
        dataloader = build_loader(
            midi_root=midi_root, csv_path=csv_path,
            seq_len=T, batch_size=base_batch_size, num_workers=num_workers,
            use_balanced_sampler=True,
        )
        counts = Counter(dataloader.dataset.labels)
        class_weights = torch.tensor(
            [1.0 / max(counts.get(c, 1), 1) for c in range(cond_dim)],
            dtype=torch.float32, device=device
        )

        # Reconfigurar CVAE para T (cambia fc_dec) y recrear su optimizador
        reconfigure_cvae_for_T(cvae, T, device)
        opt_cvae = torch.optim.Adam(cvae.parameters(), lr=5e-5)

        # Reinstanciar G/D para T, transferir pesos desde modelos previos compatibles y recrear optimizadores
        G_new, D_new = rebuild_gd_for_T(z_dim=z_dim, cond_dim=cond_dim, new_T=T, device=device)
        # transferir pesos (conv/embeds/primeras lineales)
        transfer_matching_weights(prev_G, G_new)
        transfer_matching_weights(prev_D, D_new)
        G, D = G_new, D_new
        opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # Sanity check (1 batch)
        for sample in dataloader:
            X = sample["piano_roll"].to(device).float().contiguous()
            E = sample["events"].to(device).float()
            y = sample["conditions"].to(device)
            if X.size(0) == 1:  # evitar problemas por batch=1 con cgan.py
                continue
            check_shapes_once(X, E, y, cvae, G, D, T, z_dim, cond_dim)
            break

        # Entrenar etapa
        for epoch in range(1, n_epochs + 1):
            logging.info(f"[Stage {stage}] Epoch {epoch}/{n_epochs} (batch={base_batch_size})")
            train_hybrid_epoch(cvae, G, D, dataloader, (opt_cvae, opt_g, opt_d), device, cfg, class_weights)

        # Guardar y preparar para transferir a la siguiente etapa
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(cvae.state_dict(), f"checkpoints/hybrid_cvae_T{T}.pth")
        torch.save(G.state_dict(),    f"checkpoints/hybrid_G_T{T}.pth")
        torch.save(D.state_dict(),    f"checkpoints/hybrid_D_T{T}.pth")
        logging.info(f"Guardados checkpoints para T={T}")

        prev_G, prev_D = G, D  # para transferir pesos a la siguiente etapa

    logging.info("Entrenamiento híbrido completado.")

if __name__ == "__main__":
    main()