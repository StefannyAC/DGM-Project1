# eval_elbo.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda con asistencia de ChatGPT
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este script calcula el ELBO sobre el conjunto de datos de test después de tener un modelo entrenado.
# ============================================================
import logging # logging de información
from pathlib import Path # manejo de rutas
import numpy as np # manejo de arrays
import pandas as pd # manejo de dataframes
from tqdm import tqdm # barra de progreso

import torch # Para tensores
import torch.nn.functional as F # Para funciones de activación y pérdidas

from data_pipeline import get_split_dataloader # función para obtener el split correcto 
from cvae_seq2seq import CVAE # clase CVAE 

# Configuración básica de logging: timestamp + solo mensaje
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@torch.no_grad()
def elbo_on_loader(cvae, loader, device, beta=1.0):
    """
    Función que calcula el ELBO (Evidence Lower Bound) sobre el conjunto de datos.
    ELBO = Reconstrucción - beta*KL_Divergencia
    Aquí reportamos el ELBO (maximizar) y sus componentes por separado.
    Usamos MSE como la log-verosimilitud de reconstrucción.

    Args:
        cvae: CVAE entrenada del modelo híbrido (si se quiere usar para probar la cvae pre-entrenada cambiar la pérdida de reconstrucción por BCE)
        loader: dataloader que va a cargar los datos de test
        device: 'cuda' , 'mlp', o 'cpu'
        beta: parámetro de regularización de KL 

    Returns:
        dict: Con los valores de Reconstrucción, KL y ELBO promedios obtenidos por género y en total. 
    """
    cvae.eval()
    # Usaremos diccionarios para guardar los 3 componentes
    per_gen_stats = {0: [], 1: [], 2: [], 3: []}
    all_recon, all_kl, all_elbo = [], [], []

    for batch in tqdm(loader, desc="ELBO test"):
        X = batch["piano_roll"].to(device).float()   # (B, 1, 128, T)
        E = batch["events"].to(device).float()
        y = batch["conditions"].to(device)          # (B, 1)
        y_flat = y.squeeze(-1).long()
        B = X.size(0)

        # --- Forward pass del CVAE ---
        mu, logvar = cvae.encoder(X, E, y)
        z = cvae.reparameterize(mu, logvar)
        X_recon = cvae.decode(z, y, T=X.size(-1), teacher=X)

        # --- Cálculo de los componentes del ELBO ---

        # Pérdida de Reconstrucción (BCE)
        # Lo calculamos por muestra (sumando sobre las dimensiones del piano roll)
        recon_loss = F.mse_loss(X_recon, X, reduction='none').sum(dim=(1, 2))

        # Divergencia KL
        # Fórmula estándar para D_KL(N(mu, sigma) || N(0, I))
        # Lo calculamos por muestra (sumando sobre la dimensión latente)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # ELBO
        # El loss de VAE es (Recon + beta*KL). El ELBO es el negativo de eso.
        # Queremos maximizar el ELBO (o minimizar su negativo).
        elbo = -(recon_loss + beta*kl_div)

        # --- Guardar estadísticas ---
        recon_np = recon_loss.cpu().numpy()
        kl_np = kl_div.cpu().numpy()
        elbo_np = elbo.cpu().numpy()

        all_recon.extend(recon_np.tolist())
        all_kl.extend(kl_np.tolist())
        all_elbo.extend(elbo_np.tolist())

        for i in range(B):
            gid = int(y_flat[i].item())
            if gid in per_gen_stats:
                per_gen_stats[gid].append((float(recon_np[i]), float(kl_np[i]), float(elbo_np[i])))

    # --- Agregación de resultados ---
    rows = []
    for gid in sorted(per_gen_stats.keys()):
        arr = per_gen_stats[gid]
        if not arr: continue
        a = np.array(arr)
        rows.append({
            "genre_id": gid,
            "Recon_Loss_mean": float(a[:,0].mean()),
            "KL_Div_mean": float(a[:,1].mean()),
            "ELBO_mean": float(a[:,2].mean()),
            "n": len(arr)
        })
        
    rows.append({
        "genre_id": -1,
        "Recon_Loss_mean": float(np.mean(all_recon)),
        "KL_Div_mean": float(np.mean(all_kl)),
        "ELBO_mean": float(np.mean(all_elbo)),
        "n": len(all_elbo)
    })
    return pd.DataFrame(rows)

def main():
    # --- config ---
    # Aseguramos que T coincida con el checkpoint que quieres evaluar
    T = 128 # Pues el último modelo del híbrido tiene secuencias de 128 
    batch_size = 128 # Ajustar a conveniencia
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

    # Usamos el checkpoint del CVAE del híbrido
    cvae_ckpt = pick(f"checkpoints/hybrid_cvae_best_T{T}.pth",  f"checkpoints/hybrid_cvae_T{T}_last.pth")
    assert Path(cvae_ckpt).is_file(), f"Falta checkpoint del CVAE: {cvae_ckpt}"

    # --- IMPORTANTE ---
    # La arquitectura debe coincidir EXACTAMENTE con la del checkpoint
    cvae = CVAE(z_dim=32,
        cond_dim=4,
        seq_len=T,
        ev_embed=16,
        cond_embed=4,
        enc_hid=16,
        dec_hid=16).to(device)
    
    cvae.load_state_dict(torch.load(cvae_ckpt, map_location=device))

    df = elbo_on_loader(cvae, test_loader, device, beta=0.05)
    
    out_dir = Path("eval_elbo_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"eval_elbo_T{T}.csv"
    
    df.to_csv(out_csv, index=False)
    print(df)
    print(f"Guardado: {out_csv}")

if __name__ == "__main__":
    main()