# train_cgan.py

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

from data_pipeline import build_loader
from cgan import Generator, Critic, compute_gradient_penalty    
from cvae import CVAE         

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@torch.no_grad()
def encode_with_cvae(cvae, X, E, y):
    """Retorna z del encoder de la CVAE (sin gradientes)."""
    cvae.eval()
    mu, logvar = cvae.encoder(X, E, y)
    z = cvae.reparameterize(mu, logvar)
    return z

def train_cgan_epoch(generator, critic, cvae, dataloader, opt_g, opt_c, device, config):
    generator.train(); critic.train(); cvae.eval()  # CVAE congelada aquí

    for batch in tqdm(dataloader, desc="Stage-2: Entrenando C-GAN acoplada a CVAE"):
        X  = batch["piano_roll"].to(device)     # (B, 128, T)
        E  = batch["events"].to(device)         # (B, N, 3)
        y  = batch["conditions"].to(device)     # (B, 1) género

        # --------- ENTRENAR CRÍTICO (WGAN-GP) ----------
        for _ in range(config["critic_iters"]):
            opt_c.zero_grad()

            # z desde ENCODER CVAE (congelado)
            with torch.no_grad():
                z = encode_with_cvae(cvae, X, E, y)
                X_fake = generator(z, y).detach()

            real_out = critic(X, y)
            fake_out = critic(X_fake, y)
            gp = compute_gradient_penalty(lambda x: critic(x, y), X.data, X_fake.data, device)

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

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    seq_len = 32
    batch_size = 128

    # Dataset (sin curriculum en Stage-2)
    midi_root = "dataset/data/Lakh_MIDI_Dataset_Clean"
    csv_path  = "dataset/data/lakh_clean_merged_homologado.csv"
    dataloader = build_loader(
        midi_root=midi_root,
        csv_path=csv_path,
        seq_len=seq_len,
        batch_size=batch_size,
        use_balanced_sampler=True,
    )

    # Modelos
    cvae = CVAE(z_dim=128, cond_dim=4, seq_len=seq_len).to(device)
    gen  = Generator(z_dim=128, cond_dim=4, seq_len=seq_len).to(device)
    disc = Critic(cond_dim=4, seq_len=seq_len).to(device)

    # Cargar CVAE preentrenada (Stage-1)
    ckpt_cvae = Path("checkpoints/cvae_pretrained.pth")
    assert ckpt_cvae.exists(), "Falta checkpoints/cvae_pretrained.pth (ejecuta train_cvae.py primero)."
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

    # Entrenamiento
    total_epochs = 1
    for epoch in range(total_epochs):
        logging.info(f"=== Epoch {epoch}/{total_epochs-1} (Stage-2) ===")
        train_cgan_epoch(gen, disc, cvae, dataloader, opt_g, opt_c, device, config)

    # Guardar
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(gen.state_dict(),  "checkpoints/generator_pretrained.pth")
    torch.save(disc.state_dict(), "checkpoints/critic_pretrained.pth")
    logging.info("✅ C-GAN preentrenada (acoplada a CVAE) guardada en checkpoints/")

if __name__ == "__main__":
    main()
