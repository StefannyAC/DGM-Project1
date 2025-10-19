# eval_mse.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Evalúa MSE de reconstrucción por género y global usando la CVAE
# ------------------------------------------------------------
# - Split: TEST (via data_pipeline.get_split_dataloader).
# - Pipeline por batch:
#     * Encoder: (X,E,y) → (mu, logvar) → z (reparameterize).
#     * Decoder con teacher forcing: decode(z, y, T, teacher=X).
#     * MSE por muestra (promedio en dims [128, T]); agrega por genre_id y global.
# - Checkpoints: intenta cargar "checkpoints/hybrid_cvae_best_T{T}.pth";
#   si no existe, cae a "checkpoints/hybrid_cvae_T{T}_last.pth".
# - Salida: CSV con estadísticas (mean/std) por género + fila global
#   en "eval_mse_results/eval_mse_T{T}.csv".
# - Dispositivo auto (CUDA/MPS/CPU). batch_size y T configurables.
# - No requiere el Generator; solo la CVAE.
# ============================================================

import logging # logging de información
from pathlib import Path # manejo de rutas
import numpy as np # manejo de arrays
import pandas as pd # manejo de dataframes
from tqdm import tqdm # barra de progreso

import torch # Para Tensores
import torch.nn.functional as F # Para funciones de activación y pérdidas

from data_pipeline import get_split_dataloader # Para cargar datos
from cvae_seq2seq import CVAE # Modelo CVAE

# Configuración básica de logging: timestamp + solo mensaje
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@torch.no_grad()
def mse_on_loader(cvae, loader, device):
    """
    Función para evaluar MSE por género y global usando la CVAE en un DataLoader dado.

    Para cada batch:
      - Obtiene z del encoder (mu, logvar -> reparameterize).
      - Decodifica con teacher forcing (usa X como teacher).
      - Calcula MSE por muestra (promedio sobre dims [128, T]).
      - Agrega métricas por genre_id y globales.

    Args:
        cvae: Modelo CVAE ya cargado con pesos, en el device correspondiente.
        loader: DataLoader del split objetivo (típicamente TEST).
        device: torch.device donde se ejecuta la evaluación.

    Returns:
        pd.DataFrame: Tabla con filas por genre_id (y una fila global con genre_id = -1),
        y columnas:
            - "genre_id"
            - "MSE_cvae_mean"
            - "MSE_cvae_std"
    """
    cvae.eval() # eval mode: desactiva dropout/BN training
    per_gen_stats = {0: [], 1: [], 2: [], 3: []} # acumuladores por género
    all_cvae = [] # acumulador global
    for batch in tqdm(loader, desc="MSE test"):
        X = batch["piano_roll"].to(device).float()   # (B,128,T) en [0,1]
        E = batch["events"].to(device).float()       # (B,N,3)
        y = batch["conditions"].to(device)           # (B,1)
        y_flat = y.squeeze(-1).long() # (B,) para indexar género
        B = X.size(0) # tamaño de batch

        # z del encoder (sin grad por el decorador @torch.no_grad)
        mu, logvar = cvae.encoder(X, E, y)
        z = cvae.reparameterize(mu, logvar)

        # Reconstrucción CVAE (teacher forcing), usa X como entrada del decodificador
        X_cvae = cvae.decode(z, y, T=X.size(-1), teacher=X)

        # MSE por muestra (suma sobre [128,T], promedio por B)
        mse_cvae = F.mse_loss(X_cvae, X, reduction='none').mean(dim=(1,2)).cpu().numpy()

        # Acumula global y por género
        all_cvae.extend(mse_cvae.tolist())
        for i in range(B):
            gid = int(y_flat[i].item())
            per_gen_stats[gid].append((float(mse_cvae[i])))

    # agregados por género -> filas del DataFrame
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

    cvae = CVAE(z_dim=32,
        cond_dim=4,
        seq_len=T,
        ev_embed=16,
        cond_embed=4,
        enc_hid=16,
        dec_hid=16).to(device)
    cvae.load_state_dict(torch.load(cvae_ckpt, map_location=device))

    # Evalúa MSE y guarda resultados
    df = mse_on_loader(cvae, test_loader, device)
    out_dir = Path("eval_mse_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir/f"eval_mse_T{T}.csv"
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"Guardado: {out_csv}")

if __name__ == "__main__":
    main()