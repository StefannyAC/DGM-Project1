# eval_mse.py
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data_pipeline import get_split_dataloader
from cvae_seq2seq import CVAE
from cgan import Generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@torch.no_grad()
def mse_on_loader(cvae, loader, device):
    cvae.eval()
    per_gen_stats = {0: [], 1: [], 2: [], 3: []}
    all_cvae = []
    for batch in tqdm(loader, desc="MSE test"):
        X = batch["piano_roll"].to(device).float()   # (B,128,T)
        E = batch["events"].to(device).float()
        y = batch["conditions"].to(device)           # (B,1)
        y_flat = y.squeeze(-1).long()
        B = X.size(0)

        # z del encoder
        mu, logvar = cvae.encoder(X, E, y)
        z = cvae.reparameterize(mu, logvar)

        # Reconstrucción CVAE (teacher forcing)
        X_cvae = cvae.decode(z, y, T=X.size(-1), teacher=X)

        # MSE por muestra (suma sobre [128,T], promedio por B)
        mse_cvae = F.mse_loss(X_cvae, X, reduction='none').mean(dim=(1,2)).cpu().numpy()

        all_cvae.extend(mse_cvae.tolist())
        for i in range(B):
            gid = int(y_flat[i].item())
            per_gen_stats[gid].append((float(mse_cvae[i])))

    # agregados
    rows = []
    for gid in sorted(per_gen_stats.keys()):
        arr = per_gen_stats[gid]
        if not arr: continue
        a = np.array(arr)
        rows.append({
            "genre_id": gid,
            "MSE_cvae_mean": float(a.mean()),
            "MSE_cvae_std":  float(a.std())
        })
    rows.append({
        "genre_id": -1,
        "MSE_cvae_mean": float(np.mean(all_cvae)),
        "MSE_cvae_std":  float(np.std(all_cvae)),
    })
    return pd.DataFrame(rows)

def main():
    # --- config ---
    T = 128  # longitud que quieras evaluar
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Device: {device}")

    # dataloader TEST
    test_loader = get_split_dataloader(
        seq_len=T, batch_size=batch_size, num_workers=0,
        use_balanced_sampler=False, split="test"
    )

    # carga checkpoints (elige best si existe, si no last)
    def pick(best, last):
        return best if Path(best).is_file() else last

    cvae_ckpt = pick(f"checkpoints/hybrid_cvae_best_T{T}.pth",  f"checkpoints/hybrid_cvae_T{T}_last.pth")
    assert Path(cvae_ckpt).is_file(), "Faltan checkpoints del híbrido."

    cvae = CVAE(z_dim=128, cond_dim=4, seq_len=T).to(device)
    cvae.load_state_dict(torch.load(cvae_ckpt, map_location=device))

    df = mse_on_loader(cvae, test_loader, device)
    out_dir = Path("eval_mse_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir/f"eval_mse_T{T}.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"Guardado: {out_csv}")

if __name__ == "__main__":
    main()