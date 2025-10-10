# train.py
# ============================================================
# Entrenamiento CVAE + cGAN (condicionado por género)
# con uso de piano-roll y codificación basada en eventos
# ============================================================

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_pipeline_extended import MIDIDataset  # asegúrate de usar el dataset extendido
import logging
from tqdm import tqdm
from cgan import Generator, Critic

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def get_current_seq_len(epoch: int, curriculum: dict) -> int:
    """Determina la longitud de secuencia para la época actual según el currículo."""
    sorted_epochs = sorted(curriculum.keys(), reverse=True)
    for start_epoch in sorted_epochs:
        if epoch >= start_epoch:
            return curriculum[start_epoch]
    return curriculum[min(sorted_epochs)]

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
# Entrenamiento CVAE (ELBO)
# ------------------------------------------------------------
def train_cvae_epoch(cvae_model, dataloader, cvae_optimizer, device, beta=1.0):
    """
    Entrena el CVAE solo con la pérdida ELBO.
    Usa piano_roll como entrada principal y events como información auxiliar.
    """
    cvae_model.train()
    total_recon_loss, total_kl_loss = 0, 0
    
    for batch in tqdm(dataloader, desc="Entrenando CVAE (ELBO)"):
        piano_roll = batch["piano_roll"].to(device)       # (B, 128, T)
        events = batch["events"].to(device)               # (B, N_eventos, 3)
        cond = batch["conditions"].to(device)             # (B, 1) -> género
        
        cvae_optimizer.zero_grad()
        
        # Forward: el modelo puede combinar piano_roll + events + cond
        reconstructed_batch, mu, logvar = cvae_model(piano_roll, events, cond)
        
        # ELBO = reconstrucción + divergencia KL
        recon_loss = F.binary_cross_entropy(reconstructed_batch, piano_roll, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss
        
        loss.backward()
        cvae_optimizer.step()
        
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl_loss = total_kl_loss / len(dataloader.dataset)
    logging.info(f"Pérdida CVAE -> Reconstrucción: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")



# ------------------------------------------------------------
# Entrenamiento híbrido CVAE + cGAN
# ------------------------------------------------------------
def train_hybrid_epoch(cvae, generator, critic, dataloader, optimizers, device, config):
    """
    Entrena el sistema híbrido CVAE + WGAN-GP.
    """
    cvae.train(), generator.train(), critic.train()
    opt_cvae, opt_g, opt_c = optimizers

    for batch in tqdm(dataloader, desc="Entrenando Híbrido (WGAN-GP)"):
        piano_roll = batch["piano_roll"].to(device)
        events = batch["events"].to(device)
        cond = batch["conditions"].to(device)
        
        # ----------------------------------------------------
        # A. ENTRENAR EL CRÍTICO
        # ----------------------------------------------------
        for _ in range(config['critic_iterations']):
            opt_c.zero_grad()
            
            with torch.no_grad():
                z, _, _ = cvae.encoder(piano_roll, events, cond)
                fake_sequences = generator(z, cond)
            
            real_output = critic(piano_roll, cond)
            fake_output = critic(fake_sequences, cond)
            gp = compute_gradient_penalty(lambda x: critic(x, cond), piano_roll.data, fake_sequences.data, device)
            loss_c = -torch.mean(real_output) + torch.mean(fake_output) + config['lambda_gp'] * gp
            
            loss_c.backward()
            opt_c.step()

        # ----------------------------------------------------
        # B. ENTRENAR EL GENERADOR Y EL CVAE
        # ----------------------------------------------------
        opt_g.zero_grad()
        opt_cvae.zero_grad()
        
        # --- CVAE (ELBO)
        reconstructed, mu, logvar = cvae(piano_roll, events, cond)
        recon_loss = F.mse_loss(reconstructed, piano_roll, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_cvae = recon_loss + config['beta'] * kl_loss

        # --- Generador (GAN loss)
        z, _, _ = cvae.encoder(piano_roll, events, cond)
        fake_sequences = generator(z, cond)
        fake_output = critic(fake_sequences, cond)
        loss_g = -torch.mean(fake_output)
        
        # --- Pérdida híbrida total
        loss_hybrid = config['alpha'] * loss_cvae + config['gamma'] * loss_g
        loss_hybrid.backward()
        
        opt_g.step()
        opt_cvae.step()

# ------------------------------------------------------------
# Main training controller
# ------------------------------------------------------------
def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Usando dispositivo: {device}")
    logging.info(f"INICIANDO ETAPA DE ENTRENAMIENTO: {config['training_stage']}")

    # --- Inicializa tus modelos (placeholder) ---
    # cvae = CVAEModel(...).to(device)
    generator = Generator(z_dim=128, cond_dim=4, seq_len=current_seq_len).to(device)
    critic = Critic(cond_dim=4, seq_len=current_seq_len).to(device)
    # optimizer_cvae = torch.optim.Adam(cvae.parameters(), lr=1e-4)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # optimizers = (optimizer_cvae, optimizer_g, optimizer_c)

    current_seq_len = -1
    dataloader = None
    
    for epoch in range(config['total_epochs']):
        new_seq_len = get_current_seq_len(epoch, config['curriculum'])
        
        if new_seq_len != current_seq_len:
            current_seq_len = new_seq_len
            logging.info(f"ÉPOCA {epoch}: Nueva longitud de secuencia: {current_seq_len}")
            
            dataset = MIDIDataset(midi_dir=config['midi_dir'], seq_len=current_seq_len)
            dataloader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=True,
                drop_last=True
            )
        
        logging.info(f"--- Época {epoch} ---")

        if config['training_stage'] == 1:
            # train_cvae_epoch(cvae, dataloader, optimizer_cvae, device)
            logging.info("Simulando entrenamiento de CVAE (ELBO).")
            for _ in dataloader: pass

        elif config['training_stage'] == 2:
            # train_hybrid_epoch(cvae, generator, critic, dataloader, optimizers, device, config)
            logging.info("Simulando entrenamiento híbrido CVAE + GAN.")
            for _ in dataloader: pass

        else:
            raise ValueError(f"Etapa de entrenamiento desconocida: {config['training_stage']}")
            
    logging.info("Entrenamiento finalizado.")

# ------------------------------------------------------------
# Bloque principal
# ------------------------------------------------------------
if __name__ == '__main__':
    config = {
        "training_stage": 1,
        "midi_dir": "datasets/LPD-Cleansed",
        "batch_size": 16,
        "num_workers": 0,  # en Windows usar 0
        "total_epochs": 200,
        "curriculum": {0: 32, 50: 64, 100: 128},
        
        # --- Hiperparámetros ---
        "beta": 1.0,
        "critic_iterations": 5,
        "lambda_gp": 10,
        "alpha": 1.0,
        "gamma": 1.0,
    }

    print("Lógica de entrenamiento CVAE + cGAN con events incluida. Listo para definir los modelos.")
