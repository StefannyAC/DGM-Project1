# train_cgan.py
# ============================================================
# Stage-2: Preentrenamiento de la C-GAN ACOPLADA a la CVAE
#   - z proviene del ENCODER de la CVAE (preentrenada en Stage-1)
#   - Generator(z, y) vs Critic(X, y) con WGAN-GP
# ============================================================

import torch
import logging
from tqdm import tqdm
from pathlib import Path

from data_pipeline import get_split_dataloader
from cgan import Generator, Critic, compute_gradient_penalty    
#from cvae import CVAE
from cvae_seq2seq import CVAE           

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@torch.no_grad()
def encode_with_cvae(cvae, X, E, y):
    """Retorna z del encoder de la CVAE (sin gradientes)."""
    cvae.eval()
    mu, logvar = cvae.encoder(X, E, y)
    z = cvae.reparameterize(mu, logvar)
    return z

def train_cgan_epoch(generator, critic, cvae, dataloader, opt_g, opt_c, device, config, log_every=5):
    generator.train(); critic.train(); cvae.eval()  # CVAE congelada aquí
    
    agg = {"loss_D":0.0, "loss_G":0.0, "D_real":0.0, "D_fake":0.0, "W":0.0, "GP":0.0, "steps":0}

    for batch in tqdm(dataloader, desc="Stage-2: Entrenando C-GAN acoplada a CVAE"):
        X  = batch["piano_roll"].to(device)     # (B, 128, T)
        E  = batch["events"].to(device)         # (B, N, 3)
        y  = batch["conditions"].to(device)     # (B, 1) género

        # Evitar batch=1 (por .squeeze() en el Critic/Generator condicional)
        if X.size(0) == 1:
            continue

        # --------- ENTRENAR CRÍTICO (WGAN-GP) ----------
        for _ in range(config["critic_iters"]):
            opt_c.zero_grad()

            # z desde ENCODER CVAE (congelado)
            with torch.no_grad():
                z = encode_with_cvae(cvae, X, E, y)
                X_fake = generator(z, y).detach()

            real_out = critic(X, y)
            fake_out = critic(X_fake, y)
            gp = compute_gradient_penalty(lambda x: critic(x, y), X.detach(), X_fake.detach(), device)

            loss_c = -torch.mean(real_out) + torch.mean(fake_out) + config["lambda_gp"] * gp
            loss_c.backward()
            opt_c.step()

        # ------------- ENTRENAR GENERATOR ---------------
        opt_g.zero_grad()
        # z nuevamente (sin grad al encoder)
        with torch.no_grad():
            z = encode_with_cvae(cvae, X, E, y)
        X_fake = generator(z, y)
        fake_out = critic(X_fake, y)
        loss_g = -torch.mean(fake_out)
        loss_g.backward()
        opt_g.step()

        # acumular
        D_real = real_out.mean().item()
        D_fake = fake_out.mean().item()
        Wdist  = D_real - D_fake
        GP     = gp.item()

        agg["loss_D"] += loss_c.item()
        agg["loss_G"] += loss_g.item()
        agg["D_real"] += D_real
        agg["D_fake"] += D_fake
        agg["W"]      += Wdist
        agg["GP"]     += GP
        agg["steps"]  += 1

    # promedios por epoch
    if agg["steps"] == 0:
        return None
    for k in list(agg.keys()):
        if k != "steps":
            agg[k] /= agg["steps"]
    return agg

@torch.no_grad()
def validate_cgan_epoch(generator, critic, cvae, dataloader, device, cfg):
    generator.eval(); critic.eval(); cvae.eval()

    agg = {"loss_D":0.0, "loss_G":0.0, "D_real":0.0, "D_fake":0.0, "W":0.0, "steps":0}

    for batch in tqdm(dataloader, desc="C-GAN Val", leave=False):
        X  = batch["piano_roll"].to(device)
        E  = batch["events"].to(device)
        y  = batch["conditions"].to(device)

        if X.size(0) == 1:
            continue

        z = encode_with_cvae(cvae, X, E, y)
        X_fake = generator(z, y)

        real_out = critic(X, y)
        fake_out = critic(X_fake, y)

        # Pérdidas "proxy" de validación (sin GP porque es muy costoso y ruidoso)
        loss_c = -torch.mean(real_out) + torch.mean(fake_out)
        loss_g = -torch.mean(fake_out)

        D_real = real_out.mean().item()
        D_fake = fake_out.mean().item()
        Wdist  = D_real - D_fake

        agg["loss_D"] += loss_c.item()
        agg["loss_G"] += loss_g.item()
        agg["D_real"] += D_real
        agg["D_fake"] += D_fake
        agg["W"]      += Wdist
        agg["steps"]  += 1

    if agg["steps"] == 0:
        return None
    for k in list(agg.keys()):
        if k != "steps":
            agg[k] /= agg["steps"]
    return agg

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    seq_len = 32
    batch_size = 512

    # Dataset (sin curriculum en Stage-2)
    dataloader = get_split_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        use_balanced_sampler=True,
        split="train"
    )
    val_loader = get_split_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        use_balanced_sampler=False,  # en val normalmente no balanceamos
        split="val",
    )

    # Modelos
    cvae = CVAE(z_dim=128, cond_dim=4, seq_len=seq_len).to(device)
    gen  = Generator(z_dim=128, cond_dim=4, seq_len=seq_len).to(device)
    disc = Critic(cond_dim=4, seq_len=seq_len).to(device)

    # Cargar CVAE preentrenada (Stage-1)
    ckpt_cvae = Path("checkpoints/cvae_pretrained_best.pth")
    assert ckpt_cvae.exists(), "Falta checkpoints/cvae_pretrained_best.pth (ejecuta train_cvae.py primero)."
    cvae.load_state_dict(torch.load(ckpt_cvae, map_location=device))
    for p in cvae.parameters():
        p.requires_grad_(False)  # congelada en Stage-2

    # Optimizadores
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # Config
    config = {
        "lambda_gp": 10,
        "critic_iters": 5
    }

    # ---- training loop ----
    best_val_g = float("inf")  # queremos minimizar loss_G (en validation entre más negativa = mejor)
    Path("checkpoints").mkdir(exist_ok=True)

    # Entrenamiento
    total_epochs = 5
    for epoch in range(total_epochs):
        logging.info(f"=== Epoch {epoch+1}/{total_epochs} (Stage-2) ===")
        train_stats = train_cgan_epoch(gen, disc, cvae, dataloader, opt_g, opt_c, device, config)
        if train_stats is None:
            logging.warning("Entrenamiento sin pasos válidos (¿batches de tamaño 1?).")
            continue
        logging.info(
            f"[Train] D={train_stats['loss_D']:.3f} | G={train_stats['loss_G']:.3f} | "
            f"D_real={train_stats['D_real']:.3f} D_fake={train_stats['D_fake']:.3f} | "
            f"W={train_stats['W']:.3f} | GP={train_stats['GP']:.3f} | steps={train_stats['steps']}"
        )

        val_stats = validate_cgan_epoch(gen, disc, cvae, val_loader, device, config)
        if val_stats is None:
            logging.warning("Validación sin pasos válidos.")
            continue
        logging.info(
            f"[Val]   D={val_stats['loss_D']:.3f} | G={val_stats['loss_G']:.3f} | "
            f"D_real={val_stats['D_real']:.3f} D_fake={val_stats['D_fake']:.3f} | "
            f"W={val_stats['W']:.3f}"
        )

        # guardar "best" por val loss_G
        if val_stats["loss_G"] < best_val_g:
            best_val_g = val_stats["loss_G"]
            torch.save(gen.state_dict(), "checkpoints/generator_pretrained_best.pth")
            torch.save(disc.state_dict(), "checkpoints/critic_pretrained_best.pth")
            logging.info(f"Nuevo mejor val loss_G={best_val_g:.3f} -> checkpoints/_best.pth")

    # Guardamos por si acaso la última
    torch.save(gen.state_dict(),  "checkpoints/generator_pretrained_last.pth")
    torch.save(disc.state_dict(), "checkpoints/critic_pretrained_last.pth")
    logging.info("C-GAN preentrenada (acoplada a CVAE) guardada en checkpoints/")

if __name__ == "__main__":
    main()
