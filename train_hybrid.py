# train_hybrid.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# (STAGE-3) Entrena el esquema híbrido CVAE + C-GAN
# - Usa el split de TRAIN/VAL del data_pipeline (loader balanceado opcional).
# - Carga una CVAE preentrenada (Stage-1) y la congela para extraer latentes z.
# - Alterna entrenamiento:
#     * Crítico (WGAN-GP): varias iteraciones por batch (critic_iters) con grad penalty.
#     * Generador + CVAE: pérdida híbrida  L_hyb = α·ELBO_ponderado + γ·L_G.
# - ELBO con MSE de reconstrucción + β·KL, ponderado por clase (class_weights).
# - Teacher forcing activado en la CVAE para reconstrucción durante el entrenamiento.
# - Cálculo de métricas por batch: L_D, L_G, ELBO, distancia de Wasserstein (W), GP, D_real/D_fake.
# - Promedia métricas por época y registra logs legibles.
# - Selección de mejor checkpoint por loss_G en validación; guarda también last por seguridad.
# - Dispositivo auto-seleccionado (CUDA/MPS/CPU) y carpeta 'checkpoints/' gestionada automáticamente.
# - Precaución con lotes de tamaño 1 (B=1): se omiten si el modelo no es robusto a '.squeeze()'/BatchNorm.
#
# Requisitos clave:
# - 'checkpoints/cvae_pretrained_best.pth' (producido por train_cvae.py - Stage-1).
# - data_pipeline con 'get_split_dataloader', 'collate_padded' y CSVs por split.
#
# Hiperparámetros (cfg/config):
# - lambda_gp: peso del gradient penalty (WGAN-GP).
# - critic_iters: n° de pasos del crítico por cada paso de G.
# - beta: peso del término KL en ELBO.
# - alpha, gamma: pesos de ELBO y L_G en la pérdida híbrida.
#
# Salida:
# - 'checkpoints/generator_pretrained_best.pth' y 'critic_pretrained_best.pth' (mejores por Val loss_G).
# - 'checkpoints/*_last.pth' con el último estado entrenado.
# ============================================================

import math # funciones matemáticas
import logging # logging de información
from pathlib import Path # manejo de rutas
from collections import Counter # para conteo de etiquetas

import torch # Para tensores
import torch.nn.functional as F # Para funciones de activación y pérdidas
from tqdm import tqdm # barra de progreso

from data_pipeline import get_split_dataloader # Para cargar datos
from cvae_seq2seq import CVAE # Modelo CVAE
from cgan import Generator, Critic, compute_gradient_penalty # Cargamos modelos C-GAN y GP para la pérdida

# Configuración básica de logging: timestamp + solo mensaje
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def as_long_labels(y):
    """ Garantiza que las condiciones vienen como (B,1) o (B,)"""
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y.long()

def reconfigure_cvae_for_T(cvae: CVAE, new_T: int, device: torch.device):
    """
    Ajusta el decoder del CVAE al nuevo seq_len (T).

    Args:
        cvae (CVAE): Instancia de CVAE a actualizar.
        new_T (int): Nuevo valor de longitud de secuencia (frames).
        device (torch.device): Dispositivo actual (no usado aquí, se mantiene por simetría)."""
    cvae.seq_len = new_T # actualizar atributo (seq_len) de la CVAE

def rebuild_gd_for_T(z_dim: int, cond_dim: int, new_T: int, device: torch.device):
    """
    Crea nuevas instancias de G y D para un seq_len distinto.
    Args:
        z_dim (int): Dimensionalidad de la latente 'z'.
        cond_dim (int): Número de clases de condición.
        new_T (int): Longitud de secuencia destino (frames).
        device (torch.device): Dispositivo al que enviar los modelos.

    Returns:
        tuple[nn.Module, nn.Module]: '(G_new, D_new)' reconstruidos en 'device'.
    """
    G_new = Generator(z_dim=z_dim, cond_dim=cond_dim, seq_len=new_T).to(device)
    D_new = Critic(cond_dim=cond_dim, seq_len=new_T).to(device)
    return G_new, D_new

def transfer_matching_weights(src_model: torch.nn.Module, dst_model: torch.nn.Module):
    """Copia pesos compatibles (misma clave y shape) de src a dst."""
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()
    keep = {k: v for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape}
    dst_sd.update(keep)
    dst_model.load_state_dict(dst_sd, strict=False)

def check_shapes_once(X, E, y, cvae, G, D, T, z_dim, cond_dim):
    """ Cacheo de dimensiones en los tensores"""
    B, C, TT = X.shape
    assert C == 128, f"X debe ser (B,128,T); got (B,{C},{TT})"
    assert TT == T,  f"T del batch ({TT}) != seq_len esperado ({T})"
    y_flat = as_long_labels(y)
    assert y_flat.shape[0] == B, "Batch size en y no coincide con X"
    assert y_flat.min() >= 0 and y_flat.max() < cond_dim, f"y fuera de rango [0,{cond_dim-1}]"

    with torch.no_grad():
        mu, logvar = cvae.encoder(X, E, y)
        assert mu.shape == (B, z_dim) and logvar.shape == (B, z_dim), "Encoder shapes inválidos"
        z = cvae.reparameterize(mu, logvar)
        X_rec = cvae.decode(z, y, T = T, teacher = None)
        assert X_rec.shape == (B, 128, T), f"Decoder shape inválido: {tuple(X_rec.shape)}!=(B,128,{T})"
        X_fake = G(z, y)
        r_out = D(X, y)
        f_out = D(X_fake, y)
        assert r_out.shape[0] == B and f_out.shape[0] == B, "Critic batch mismatch"

def train_hybrid_epoch(cvae, G, D, dataloader, opts, device, cfg, class_weights):
    """
    Función para entrenar 1 época del modelo híbrido CVAE + C-GAN.
    - Actualiza primero el Crítico (WGAN-GP) varias iteraciones por batch.
    - Luego actualiza Generador (G) y CVAE con una pérdida compuesta:
        L_hyb = α * ELBO_ponderado + γ * L_G

    Args:
        cvae: Modelo CVAE (provee reconstrucción y latentes z).
        G: Generador condicional de la cGAN (produce X_fake a partir de z, y).
        D: Crítico/Discriminador condicional (para WGAN-GP).
        dataloader: Iterador de batches con claves {"piano_roll","events","conditions"}.
        opts: Tupla de optimizadores (opt_cvae, opt_g, opt_d) en ese orden.
        device: Dispositivo ('cuda'/'mps'/'cpu').
        cfg (dict): Hiperparámetros (requiere: "critic_iters", "lambda_gp", "beta", "alpha", "gamma").
        class_weights (torch.Tensor): Pesos por clase (longitud = n_clases) para ponderar ELBO.

    Returns:
        dict: Promedios por época de:
            {"L_hyb","ELBO","L_G","L_D","D_real","D_fake","W","GP","steps"}
            o 'None' si no hubo pasos válidos.
    """
    cvae.train(); G.train(); D.train() # Modo entrenamiento para los tres modelos
    opt_cvae, opt_g, opt_d = opts # Desempaqueta optimizadores

    # Acumuladores por época
    total_hyb, total_elbo, total_lg, total_ld = 0.0, 0.0, 0.0, 0.0
    total_Dreal, total_Dfake, total_W, total_GP = 0.0, 0.0, 0.0, 0.0
    n_steps = 0

    for batch in tqdm(dataloader, desc="Entrenando híbrido CVAE + C-GAN"):
        X = batch["piano_roll"].to(device).float().contiguous()  # (B,128,T) reales
        E = batch["events"].to(device).float() # (B,N,3)
        y = batch["conditions"].to(device) # (B,1)
        y_flat = as_long_labels(y) # (B,) normaliza condición para indexar pesos

        # Evitar batch=1 (por cgan.py 'squeeze()' sin eje)
        if X.size(0) == 1:
            continue

        # ----- Critic (WGAN-GP) -----
        for _ in range(cfg["critic_iters"]):
            opt_d.zero_grad()

            # Congelamos CVAE y G aquí: solo queremos actualizar D
            with torch.no_grad():
                mu, logvar = cvae.encoder(X, E, y) # parámetros q(z|X,E,y)
                z = cvae.reparameterize(mu, logvar) # muestreo latente
                X_fake = G(z, y).detach() # fakes (no grad a G)

            real_out = D(X, y).squeeze(1) # D(x_real, y)
            fake_out = D(X_fake, y).squeeze(1) # D(x_fake, y)

            # Penalización de gradiente para imponer ||∇_x D||->1
            gp = compute_gradient_penalty(lambda t: D(t, y), X.detach(), X_fake.detach(), device)

            # L_D = E[D(fake)] - E[D(real)] + lambda*GP  (equivalente a -Wdist + lambda*GP)
            loss_d = -torch.mean(real_out) + torch.mean(fake_out) + cfg["lambda_gp"] * gp
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0) # clip defensivo
            opt_d.step()

            # Métricas del crítico (de la última iter de critic en este batch)
            D_real = real_out.mean().item()
            D_fake = fake_out.mean().item()
            Wdist  = D_real - D_fake
            GP     = gp.item()

        # ----- Generator + CVAE -----
        opt_g.zero_grad()
        opt_cvae.zero_grad()

        # ELBO (MSE) con ponderación por clase: recon + β * KL, ponderado por w[y]
        X_rec, mu, logvar = cvae(X, E, y, teacher_prob=1.0) # reconstrucción con teacher forcing
        recon = F.mse_loss(X_rec, X, reduction='none').sum(dim=(1,2)) # ||X-X_rec||^2 por muestra
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1) # KL(q||p) por muestra

        w = class_weights[y_flat] # peso por clase para cada muestra
        elbo = (w * (recon + cfg["beta"] * kl)).mean() # promedio ponderado

        # Componente adversaria para G (y CVAE vía z): L_G = -E[D(G(z),y)]
        with torch.no_grad():
            z = cvae.reparameterize(mu, logvar) # latentes (sin grad a encoder)
        X_fake = G(z, y)
        g_out = D(X_fake, y).squeeze(1)
        loss_g = -torch.mean(g_out)

        # Pérdida híbrida total: α*ELBO + γ*L_G
        loss_hyb = cfg["alpha"] * elbo + cfg["gamma"] * loss_g
        loss_hyb.backward()
        torch.nn.utils.clip_grad_norm_(cvae.parameters(), 1.0) # clips defensivos
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0) # clips defensivos
        opt_g.step()
        opt_cvae.step()

        # acumular (por batch)
        total_hyb += loss_hyb.item()
        total_elbo += elbo.item()
        total_lg += loss_g.item()
        total_ld += loss_d.item()
        total_Dreal += D_real
        total_Dfake += D_fake
        total_W += Wdist
        total_GP += GP
        n_steps += 1

        # checks simples (heurística para detectar posible colapso/estancamiento)
        if abs(Wdist) < 0.05 and loss_g.item() < 0.01:
            logging.debug("Posible estancamiento: |W|->0 y loss_g->0 en este batch.")

    # promedios por época
    if n_steps == 0:
        return None
    return {
        "L_hyb": total_hyb / n_steps,
        "ELBO": total_elbo / n_steps,
        "L_G": total_lg / n_steps,
        "L_D": total_ld / n_steps,
        "D_real": total_Dreal / n_steps,
        "D_fake": total_Dfake / n_steps,
        "W": total_W / n_steps,
        "GP": total_GP / n_steps,
        "steps": n_steps
    }

@torch.no_grad()
def validate_hybrid_epoch(cvae, G, D, dataloader, device, cfg, class_weights):
    """Validación: sin updates. Usamos proxy L_D (real-fake) y L_G exactamente igual; sin GP."""
    cvae.eval(); G.eval(); D.eval()

    total_hyb = total_elbo = total_lg = total_ld = 0.0
    total_Dreal = total_Dfake = total_W = 0.0
    n_steps = 0

    for batch in tqdm(dataloader, desc="Hybrid Val", leave=False):
        X = batch["piano_roll"].to(device).float().contiguous()
        E = batch["events"].to(device).float()
        y = batch["conditions"].to(device)
        y_flat = as_long_labels(y)

        if X.size(0) == 1:
            continue

        # ELBO con teacher forcing para medir reconstrucción limpia
        X_rec, mu, logvar = cvae(X, E, y, teacher_prob=1.0)
        recon = F.mse_loss(X_rec, X, reduction='none').sum(dim=(1,2))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        w = class_weights[y_flat]
        elbo = (w * (recon + cfg["beta"] * kl)).mean()

        z = cvae.reparameterize(mu, logvar)
        X_fake = G(z, y)
        real_out = D(X, y).squeeze(1)
        fake_out = D(X_fake, y).squeeze(1)

        # proxy de D (sin GP)
        loss_d = -torch.mean(real_out) + torch.mean(fake_out)
        loss_g = -torch.mean(fake_out)

        loss_hyb = cfg["alpha"] * elbo + cfg["gamma"] * loss_g

        D_real = real_out.mean().item()
        D_fake = fake_out.mean().item()
        Wdist  = D_real - D_fake

        total_hyb += loss_hyb.item()
        total_elbo += elbo.item()
        total_lg += loss_g.item()
        total_ld += loss_d.item()
        total_Dreal += D_real
        total_Dfake += D_fake
        total_W += Wdist
        n_steps += 1

    if n_steps == 0:
        return None
    return {
        "L_hyb": total_hyb / n_steps,
        "ELBO": total_elbo / n_steps,
        "L_G": total_lg / n_steps,
        "L_D": total_ld / n_steps,
        "D_real": total_Dreal / n_steps,
        "D_fake": total_Dfake / n_steps,
        "W": total_W / n_steps,
        "steps": n_steps
    }


def main():
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"\nUsing Device: {device}\n")

    # --- config ---
    curriculum = [32, 64, 128]   # seq_len reales por etapa
    epochs_per = [10, 10, 15]       # ajusta a gusto, dejar más épocas el de 128

    z_dim = 32
    cond_dim = 4
    ev_embed = 16
    cond_embed = 4

    base_batch_size = 512   # <- batch fijo
    num_workers = 0

    # modelos (arrancan en la primera T del currículo)
    cvae = CVAE(
        z_dim=z_dim,
        cond_dim=cond_dim,
        seq_len=curriculum[0],
        ev_embed=ev_embed,
        cond_embed=cond_embed,
        enc_hid=16,
        dec_hid=16,
    ).to(device)
    G = Generator(
        z_dim=z_dim,
        cond_dim=cond_dim,
        seq_len=curriculum[0],
        hidden_dim=32
    ).to(device)
    D = Critic(
        cond_dim=cond_dim,
        seq_len=curriculum[0],
        input_dim=128,
        hidden_dim=32
    ).to(device)

    # Carga preentrenos (deben existir)
    ckpt_cvae = Path("checkpoints/cvae_pretrained_best.pth")
    if ckpt_cvae.exists():
        cvae.load_state_dict(torch.load(ckpt_cvae, map_location=device))
        logging.info("CVAE preentrenada cargada.")
    ckpt_g = Path("checkpoints/generator_pretrained_best.pth")
    if ckpt_g.exists():
        G.load_state_dict(torch.load(ckpt_g, map_location=device))
        logging.info("Generator preentrenado cargado.")
    ckpt_d = Path("checkpoints/critic_pretrained_best.pth")
    if ckpt_d.exists():
        D.load_state_dict(torch.load(ckpt_d, map_location=device))
        logging.info("Critic preentrenado cargado.")

    # Opts iniciales
    opt_cvae = torch.optim.Adam(cvae.parameters(), lr=5e-5)
    opt_g    = torch.optim.Adam(G.parameters(),    lr=1e-4, betas=(0.5, 0.999))
    opt_d    = torch.optim.Adam(D.parameters(),    lr=1e-4, betas=(0.5, 0.999))

    cfg = {
        "beta": 0.01,
        "alpha": 1.0,
        "gamma": 0.5,
        "critic_iters": 5,
        "lambda_gp": 10.0,
        "recon_loss": "mse",
    }

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    logging.info("=== Iniciando entrenamiento híbrido CVAE + C-GAN con currículum ===")
    # === Curriculum real: 32 -> 64 -> 128 ===
    # T representa la longitud temporal (seq_len)
    # En cada etapa se reconfiguran CVAE, G y D para el nuevo T
    prev_G, prev_D = G, D  # para transferir pesos entre etapas
    for stage, (T, n_epochs) in enumerate(zip(curriculum, epochs_per), 1):
        logging.info(f"\n===== CURRICULUM STAGE {stage}: seq_len={T}, epochs={n_epochs} =====")

        # DataLoader para el nuevo T
        dataloader = get_split_dataloader(
            seq_len=T, batch_size=base_batch_size, num_workers=num_workers,
            use_balanced_sampler=True, split='train'
        )
        val_loader = get_split_dataloader(
            seq_len=T, batch_size=base_batch_size, num_workers=num_workers,
            use_balanced_sampler=False, split='val'
        )

        counts = Counter(dataloader.dataset.labels)
        class_weights = torch.tensor(
            [1.0 / max(counts.get(c, 1), 1) for c in range(cond_dim)],
            dtype=torch.float32, device=device
        )

        # Reconfigurar CVAE para T (cambia fc_dec) y recrear su optimizador
        reconfigure_cvae_for_T(cvae, T, device)
        opt_cvae = torch.optim.Adam(cvae.parameters(), lr=5e-5)

        # Reinstanciar G/D para T, transferir pesos desde modelos previos compatibles y recrear optimizadores
        G_new, D_new = rebuild_gd_for_T(z_dim=z_dim, cond_dim=cond_dim, new_T=T, device=device)
        # transferir pesos (conv/embeds/primeras lineales)
        transfer_matching_weights(prev_G, G_new)
        transfer_matching_weights(prev_D, D_new)
        G, D = G_new, D_new
        opt_g = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # Sanity check (1 batch)
        for sample in dataloader:
            X = sample["piano_roll"].to(device).float().contiguous()
            E = sample["events"].to(device).float()
            y = sample["conditions"].to(device)
            if X.size(0) == 1:  # evitar problemas por batch=1 con cgan.py
                continue
            check_shapes_once(X, E, y, cvae, G, D, T, z_dim, cond_dim)
            break

        # early best-tracking por etapa
        best_val = float('inf')
        best_tag = f"_best_T{T}.pth"

        # Entrenar etapa
        for epoch in range(1, n_epochs + 1):
            logging.info(f"[Stage {stage}] Epoch {epoch}/{n_epochs} (batch={base_batch_size})")
            train_stats = train_hybrid_epoch(cvae, G, D, dataloader, (opt_cvae, opt_g, opt_d), device, cfg, class_weights)
            if train_stats is None:
                logging.warning("Sin pasos válidos (quizá batch=1 todos los lotes).")
                continue
            logging.info(
                f"[Train] L_hyb={train_stats['L_hyb']:.4f} | ELBO={train_stats['ELBO']:.4f} | "
                f"L_G={train_stats['L_G']:.4f} | L_D={train_stats['L_D']:.4f} | "
                f"D_real={train_stats['D_real']:.3f} D_fake={train_stats['D_fake']:.3f} | "
                f"W={train_stats['W']:.3f} | GP={train_stats['GP']:.3f} | steps={train_stats['steps']}"
            )

            val_stats = validate_hybrid_epoch(cvae, G, D, val_loader, device, cfg, class_weights)
            if val_stats is None:
                logging.warning("Validación sin pasos válidos.")
                continue
            logging.info(
                f"[Val]   L_hyb={val_stats['L_hyb']:.4f} | ELBO={val_stats['ELBO']:.4f} | "
                f"L_G={val_stats['L_G']:.4f} | L_D={val_stats['L_D']:.4f} | "
                f"D_real={val_stats['D_real']:.3f} D_fake={val_stats['D_fake']:.3f} | "
                f"W={val_stats['W']:.3f} | steps={val_stats['steps']}"
            )

            # Guardar "best" por val L_hyb
            if val_stats["L_hyb"] < best_val:
                best_val = val_stats["L_hyb"]
                torch.save(cvae.state_dict(), f"checkpoints/hybrid_cvae{best_tag}")
                torch.save(G.state_dict(),    f"checkpoints/hybrid_G{best_tag}")
                torch.save(D.state_dict(),    f"checkpoints/hybrid_D{best_tag}")
                logging.info(f"Nuevo mejor Val L_hyb={best_val:.4f} (T={T})")


        # Guardar y preparar para transferir a la siguiente etapa
        torch.save(cvae.state_dict(), f"checkpoints/hybrid_cvae_T{T}_last.pth")
        torch.save(G.state_dict(),    f"checkpoints/hybrid_G_T{T}_last.pth")
        torch.save(D.state_dict(),    f"checkpoints/hybrid_D_T{T}_last.pth")
        logging.info(f"Guardados checkpoints (last) para T={T}")

        prev_G, prev_D = G, D  # para transferir pesos a la siguiente etapa

    logging.info("Entrenamiento híbrido completado.")

if __name__ == "__main__":
    main()