# train_cvae.py
import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm

from data_pipeline import get_split_dataloader
#from cvae import CVAE
from cvae_seq2seq import CVAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def train_cvae_epoch(model, dataloader, optimizer, device, beta=1.0, class_weights=None):
    model.train()
    total_recon, total_kl, total_samples = 0.0, 0.0, 0

    for batch in tqdm(dataloader, desc="Entrenando CVAE"):
        X = batch["piano_roll"].to(device).float()   # (B,128,T) en [0,1]
        E = batch["events"].to(device).float()       # (B,N,3)
        y = batch["conditions"].to(device)           # (B,1) long
        B = X.size(0)

        optimizer.zero_grad()
        X_rec, mu, logvar = model(X, E, y)

        # --- pérdidas por muestra ---
        # BCE por muestra
        bce = F.binary_cross_entropy(X_rec, X, reduction='none')  # (B,128,T)
        bce = bce.sum(dim=(1,2))  # (B,)

        # KL por muestra
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)  # (B,)

        if class_weights is not None:
            # y: (B,1) -> (B,)
            y_flat = y.squeeze(-1).long()
            w = class_weights[y_flat]  # (B,)
            loss = (w * (bce + beta * kl)).mean()
        else:
            loss = (bce + beta * kl).mean()

        loss.backward()
        optimizer.step()

        total_recon += bce.sum().item()
        total_kl    += kl.sum().item()
        total_samples += B

    logging.info(
        f"Recon/sample: {total_recon/total_samples:.2f} | "
        f"KL/sample: {total_kl/total_samples:.2f}"
    )

def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\nUsing Device: {device}\n")
    seq_len   = 32
    batch_size = 512
    num_workers = 0  # Windows -> 0

    # loader (balanceado por defecto)
    dataloader = get_split_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        num_workers=num_workers,
        use_balanced_sampler=True,
        split="train"
    )

    # pesos por clase para ELBO (inverso de frecuencia)
    from collections import Counter
    counts = Counter(dataloader.dataset.labels)
    w_vec = torch.tensor(
        [1.0 / max(counts.get(c, 1), 1) for c in range(4)],
        dtype=torch.float32, device=device
    )

    model = CVAE(z_dim=128, cond_dim=4, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # estabilidad

    epochs = 5
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        train_cvae_epoch(model, dataloader, optimizer, device, beta=1.0, class_weights=w_vec)

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/cvae_pretrained.pth")
    logging.info("CVAE preentrenado guardado en checkpoints/cvae_pretrained.pth")

if __name__ == "__main__":
    from pathlib import Path
    main()