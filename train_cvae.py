#train_cvae.py

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
from tqdm import tqdm
from data_pipeline import MIDIDataset
from cvae import CVAE
from utils_collate import collate_padded

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_cvae_epoch(model, dataloader, optimizer, device, beta=1.0):
    model.train()
    total_recon, total_kl = 0, 0
    for batch in tqdm(dataloader, desc="Entrenando CVAE"):
        piano_roll = batch["piano_roll"].to(device)
        events = batch["events"].to(device)
        cond = batch["conditions"].to(device)

        optimizer.zero_grad()
        reconstructed, mu, logvar = model(piano_roll, events, cond)

        recon_loss = F.binary_cross_entropy(reconstructed, piano_roll, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        loss.backward()
        optimizer.step()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    logging.info(f"Recon: {total_recon/len(dataloader.dataset):.4f}, KL: {total_kl/len(dataloader.dataset):.4f}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = MIDIDataset("datasets/LPD-Cleansed", seq_len=128)
    dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0, 
    collate_fn=collate_padded)

    model = CVAE(z_dim=128, cond_dim=4, seq_len=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(100):
        train_cvae_epoch(model, dataloader, optimizer, device)

    torch.save(model.state_dict(), "checkpoints/cvae_pretrained.pth")
    logging.info("CVAE preentrenado guardado en checkpoints/")

if __name__ == "__main__":
    main()
