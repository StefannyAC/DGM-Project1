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
    
    tot_ld = tot_lg = tot_dreal = tot_dfake = tot_w = tot_gp = 0.0
    steps = 0

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
        GP     = float(gp.item())

        tot_ld += loss_c.item(); tot_lg += loss_g.item()
        tot_dreal += D_real; tot_dfake += D_fake
        tot_w += Wdist; tot_gp += GP
        steps += 1

    # promedios por epoch
    return {
        "loss_D": tot_ld/steps, "loss_G": tot_lg/steps,
        "D_real": tot_dreal/steps, "D_fake": tot_dfake/steps,
        "W": tot_w/steps, "GP": tot_gp/steps, "steps": steps
    }

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
    total_epochs = 5
    for epoch in range(total_epochs):
        logging.info(f"=== Epoch {epoch+1}/{total_epochs} (Stage-2) ===")
        stats = train_cgan_epoch(gen, disc, cvae, dataloader, opt_g, opt_c, device, config)
        logging.info(
            f"[Epoch {epoch+1}/{total_epochs}] "
            f"D={stats['loss_D']:.3f} | G={stats['loss_G']:.3f} | "
            f"D_real={stats['D_real']:.3f} D_fake={stats['D_fake']:.3f} | "
            f"W={stats['W']:.3f} | GP={stats['GP']:.3f} | steps={stats['steps']}"
        )

    # Guardar
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(gen.state_dict(),  "checkpoints/generator_pretrained.pth")
    torch.save(disc.state_dict(), "checkpoints/critic_pretrained.pth")
    logging.info("C-GAN preentrenada (acoplada a CVAE) guardada en checkpoints/")

if __name__ == "__main__":
    main()
