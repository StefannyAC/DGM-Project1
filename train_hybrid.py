# train_hybrid.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from tqdm import tqdm
from data_pipeline import MIDIDataset
from cvae import CVAE
from cgan import Generator, Critic, compute_gradient_penalty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Helper function
def get_current_seq_len(epoch: int, curriculum: dict) -> int:
    """Determina la longitud de secuencia para la época actual según el currículo."""
    sorted_epochs = sorted(curriculum.keys(), reverse=True)
    for start_epoch in sorted_epochs:
        if epoch >= start_epoch:
            return curriculum[start_epoch]
    return curriculum[min(sorted_epochs)]

# Entrenamiento híbrido CVAE + C-GAN
def train_hybrid_epoch(cvae, generator, critic, dataloader, optimizers, device, config):
    cvae.train(); generator.train(); critic.train()
    opt_cvae, opt_g, opt_c = optimizers

    for batch in tqdm(dataloader, desc="Entrenando híbrido CVAE + C-GAN"):
        piano_roll = batch["piano_roll"].to(device)
        events = batch["events"].to(device)
        cond = batch["conditions"].to(device)

        # --- Critic ---
        for _ in range(config["critic_iters"]):
            opt_c.zero_grad()
            z, _, _ = cvae.encoder(piano_roll, events, cond)
            fake = generator(z, cond)
            real_out = critic(piano_roll, cond)
            fake_out = critic(fake.detach(), cond)
            gp = compute_gradient_penalty(lambda x: critic(x, cond), piano_roll.data, fake.data, device)
            loss_c = -torch.mean(real_out) + torch.mean(fake_out) + config["lambda_gp"] * gp
            loss_c.backward(); opt_c.step()

        # --- Generator + CVAE ---
        opt_g.zero_grad(); opt_cvae.zero_grad()
        reconstructed, mu, logvar = cvae(piano_roll, events, cond)
        recon_loss = F.mse_loss(reconstructed, piano_roll, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_cvae = recon_loss + config["beta"] * kl_loss
        z, _, _ = cvae.encoder(piano_roll, events, cond)
        fake = generator(z, cond)
        fake_out = critic(fake, cond)
        loss_g = -torch.mean(fake_out)
        loss_total = config["alpha"] * loss_cvae + config["gamma"] * loss_g
        loss_total.backward()
        opt_g.step(); opt_cvae.step()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = MIDIDataset("datasets/LPD-Cleansed", seq_len=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Cargar modelos preentrenados
    cvae = CVAE(z_dim=128, cond_dim=4, seq_len=128).to(device)
    generator = Generator(z_dim=128, cond_dim=4, seq_len=128).to(device)
    critic = Critic(cond_dim=4, seq_len=128).to(device)
    cvae.load_state_dict(torch.load("checkpoints/cvae_pretrained.pth"))
    generator.load_state_dict(torch.load("checkpoints/generator_pretrained.pth"))
    critic.load_state_dict(torch.load("checkpoints/critic_pretrained.pth"))

    opt_cvae = torch.optim.Adam(cvae.parameters(), lr=1e-5)
    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-5, betas=(0.5, 0.999))

    config = {"alpha":1.0, "beta":1.0, "gamma":0.5, "lambda_gp":10, "critic_iters":5, "total_epochs": 200,"batch_size": 16,"curriculum": {0: 32, 50: 64, 100: 128}}

    current_seq_len = -1
    for epoch in range(config["total_epochs"]):
        new_seq_len = get_current_seq_len(epoch, config["curriculum"])
        if new_seq_len != current_seq_len:
            current_seq_len = new_seq_len
            logging.info(f"[Curriculum] Epoch {epoch}: secuencia = {current_seq_len}")

            dataset = MIDIDataset("datasets/LPD-Cleansed", seq_len=current_seq_len)
            dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        train_hybrid_epoch(cvae, generator, critic, dataloader,
                        (opt_cvae, opt_g, opt_c), device, config)

    torch.save(cvae.state_dict(), "checkpoints/cvae_finetuned.pth")
    torch.save(generator.state_dict(), "checkpoints/generator_finetuned.pth")
    torch.save(critic.state_dict(), "checkpoints/critic_finetuned.pth")
    logging.info("Entrenamiento híbrido completado y guardado.")

if __name__ == "__main__":
    main()