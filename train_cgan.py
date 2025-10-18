# train_cgan.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este script implementa el Stage-2 del entrenamiento, el cual corresponde a entrenar una C-GAN
# acoplada a una CVAE preentrenada (Stage-1 en train_cvae.py), haciendo uso del espacio lantente 
# de la CVAE. Y utiliza WGAN-GP como técnica de estabilización. Además, incluye funciones de
# entrenamiento y validación por epoch, así como el guardado del mejor modelo basado en la
# pérdida de validación del generador. Requiere data_pipeline.py, cvae_seq2seq.py y cgan.py.
# ============================================================
# Stage-2: Preentrenamiento de la C-GAN ACOPLADA a la CVAE
#   - z proviene del ENCODER de la CVAE (preentrenada en train_cvae.py)
#   - Generator(z, y) vs Critic(X, y) con WGAN-GP
# ============================================================

import torch # Para tensores
import logging # Para logging de información
from tqdm import tqdm # Para barras de progreso
from pathlib import Path # Para manejo de rutas

from data_pipeline import get_split_dataloader # Para cargar datos
from cgan import Generator, Critic, compute_gradient_penalty # Modelos C-GAN y GP
from cvae_seq2seq import CVAE # Modelo CVAE

# Configuración básica de logging: timestamp + solo mensaje
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@torch.no_grad()
def encode_with_cvae(cvae, X, E, y):
    """
    Función para obtener la latente z de la CVAE sin gradientes (modo evaluación).

    Args:
        cvae (CVAE): Modelo CVAE ya inicializado (se asume preentrenado).
        X: Secuencias de entrada (B, 128, T) para el encoder.
        E: Eventos (B, N, 3) que el EventEncoder de la CVAE resumirá.
        y: Condición categórica (B,) o (B,1).

    Returns:
        torch.Tensor: Latentes z muestreadas por reparametrización (B, z_dim).
    """
    cvae.eval() # desactiva dropout/BN y usa stats fijas
    mu, logvar = cvae.encoder(X, E, y) # parámetros de q(z|X,E,y)
    z = cvae.reparameterize(mu, logvar) # z = mu + sigma * eps (sin gradientes)
    return z

def train_cgan_epoch(generator, critic, cvae, dataloader, opt_g, opt_c, device, config):
    """
    Función para entrenar una época de la C-GAN (WGAN-GP) acoplada a una CVAE congelada.
    Entrena primero el Crítico varias veces por lote (WGAN-GP), luego actualiza el Generador.
    La CVAE solo provee z (fija/congelada en Stage-2).

    Args:
        generator (Generator): Generador condicional G(z, y) -> X_fake.
        critic (Critic): Crítico/Discriminador condicional D(X, y).
        cvae (CVAE): CVAE preentrenada (solo para producir z del encoder).
        dataloader: Iterador de lotes de entrenamiento.
        opt_g: Optimizador de G.
        opt_c: Optimizador de D/Critic.
        device: 'cuda' | 'mps' | 'cpu'.
        config: Hiperparámetros, requiere:
            - "lambda_gp": peso del término de grad penalty.
            - "critic_iters": pasos del crítico por cada paso de G.

    Returns:
        dict: Promedios de pérdidas/estadísticos por época.
    """
    generator.train(); critic.train(); cvae.eval()  # CVAE se usa congelada (no se actualiza)
    
    # Acumuladores de métricas por época
    agg = {"loss_D":0.0, "loss_G":0.0, "D_real":0.0, "D_fake":0.0, "W":0.0, "GP":0.0, "steps":0}

    # Bucle por lotes
    for batch in tqdm(dataloader, desc="Stage-2: Entrenando C-GAN acoplada a CVAE"):
        X  = batch["piano_roll"].to(device)     # (B, 128, T) entrada real
        E  = batch["events"].to(device)         # (B, N, 3) eventos para encoder CVAE
        y  = batch["conditions"].to(device)     # (B, 1) género condicional

        # Evitar batch=1 (por .squeeze() en el Critic/Generator condicional)
        if X.size(0) == 1:
            continue

        # --------- ENTRENAR CRÍTICO (WGAN-GP) ----------
        for _ in range(config["critic_iters"]):
            opt_c.zero_grad() # gradientes del crítico a cero

            # z desde ENCODER CVAE (congelado) y un fake con G (detach para no backprop a G)
            with torch.no_grad():
                z = encode_with_cvae(cvae, X, E, y) # (B, z_dim)
                X_fake = generator(z, y).detach() # (B, 128, T)

            # Puntajes del crítico
            real_out = critic(X, y) # D(x_real, y)
            fake_out = critic(X_fake, y) # D(x_fake, y)

            # Gradient penalty (WGAN-GP) entre real y fake
            gp = compute_gradient_penalty(lambda x: critic(x, y), X.detach(), X_fake.detach(), device)

            # Pérdida del crítico WGAN-GP E[D(fake)] - E[D(real)] + lambda * GP (equivalente a -Wdist + lambda*GP)
            loss_c = -torch.mean(real_out) + torch.mean(fake_out) + config["lambda_gp"] * gp
            loss_c.backward() # backprop sobre parámetros del crítico
            opt_c.step() # actualización del crítico

        # ------------- ENTRENAR GENERATOR ---------------
        opt_g.zero_grad() # gradientes del generador a cero
        # z nuevamente (no queremos gradiente hacia el encoder de la CVAE)
        with torch.no_grad():
            z = encode_with_cvae(cvae, X, E, y)
        X_fake = generator(z, y) # genera nuevas muestras
        fake_out = critic(X_fake, y) # evalúa el crítico sobre fake

        # Pérdida de G en WGAN: -E[D(fake)]  (maximiza D(fake))
        loss_g = -torch.mean(fake_out)
        loss_g.backward() # backprop en G
        opt_g.step() # actualización de G

        # --------- Métricas y acumulación ----------
        D_real = real_out.mean().item() # promedio del batch (reales)
        D_fake = fake_out.mean().item() # promedio del batch (fakes)
        Wdist  = D_real - D_fake # aproximación a la distancia de Wasserstein
        GP     = gp.item()

        agg["loss_D"] += loss_c.item()
        agg["loss_G"] += loss_g.item()
        agg["D_real"] += D_real
        agg["D_fake"] += D_fake
        agg["W"]      += Wdist
        agg["GP"]     += GP
        agg["steps"]  += 1

    # promedios por época (si hubo pasos válidos)
    if agg["steps"] == 0:
        return None
    for k in list(agg.keys()):
        if k != "steps":
            agg[k] /= agg["steps"] # promedios por cantidad de pasos
    return agg

@torch.no_grad() # sin gradientes en validación
def validate_cgan_epoch(generator, critic, cvae, dataloader, device):
    """
    Función para validar una época de la C-GAN (sin actualizar pesos).
    Usa pérdidas proxy (sin grad penalty) para reducir costo y ruido en validación.

    Args:
        generator (Generator): Generador condicional G.
        critic (Critic): Crítico condicional D.
        cvae (CVAE): CVAE preentrenada (solo para producir z).
        dataloader: Iterador de lotes de validación.
        device: 'cuda' | 'mps' | 'cpu'.

    Returns:
        dict: Promedios de {"loss_D","loss_G","D_real","D_fake","W","steps"}.
    """
    generator.eval(); critic.eval(); cvae.eval() # todo en modo eval (sin dropout/BN training)

    # Acumuladores de métricas por época
    agg = {"loss_D":0.0, "loss_G":0.0, "D_real":0.0, "D_fake":0.0, "W":0.0, "steps":0}

    # Bucle por lotes
    for batch in tqdm(dataloader, desc="C-GAN Val", leave=False):
        X  = batch["piano_roll"].to(device) # (B, 128, T) entrada real
        E  = batch["events"].to(device) # (B, N, 3) eventos para encoder CVAE
        y  = batch["conditions"].to(device) # (B, 1) género condicional

        if X.size(0) == 1:
            continue

        z = encode_with_cvae(cvae, X, E, y) # (B, z_dim)
        X_fake = generator(z, y) # (B, 128, T)

        real_out = critic(X, y) # D(x_real, y)
        fake_out = critic(X_fake, y) # D(x_fake, y)

        # Pérdidas "proxy" de validación (sin GP porque es muy costoso y ruidoso),  E[D(fake)] - E[D(real)]  y  -E[D(fake)]
        loss_c = -torch.mean(real_out) + torch.mean(fake_out) # pérdida crítico proxy
        loss_g = -torch.mean(fake_out) # pérdida generador proxy

        # Métricas y acumulación
        D_real = real_out.mean().item() # promedio del batch (reales)
        D_fake = fake_out.mean().item() # promedio del batch (fakes)
        Wdist  = D_real - D_fake # aproximación a la distancia de Wasserstein

        # Acumular
        agg["loss_D"] += loss_c.item()
        agg["loss_G"] += loss_g.item()
        agg["D_real"] += D_real
        agg["D_fake"] += D_fake
        agg["W"]      += Wdist
        agg["steps"]  += 1

    if agg["steps"] == 0:
        return None
    for k in list(agg.keys()):
        if k != "steps":
            agg[k] /= agg["steps"] # promedios por cantidad de pasos
    return agg

def main():
    """
    Función para ejecutar el pipeline de Stage-2:
    carga datos, arma modelos (CVAE+GAN), carga pesos de la CVAE,
    entrena por épocas (WGAN-GP) y guarda mejores checkpoints.
    """
    # ---- Configuración inicial ----
    # Selección de dispositivo
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    seq_len = 32 # longitud de secuencia (igual que en Stage-1)
    batch_size = 512 # tamaño de lote

    # ------------------ DataLoaders ------------------
    # Entrenamiento (con muestreador balanceado por clase)
    dataloader = get_split_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        use_balanced_sampler=True,
        split="train"
    )
    # Validación (sin balanceo para reflejar distribución real)
    val_loader = get_split_dataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        use_balanced_sampler=False,  # en val normalmente no balanceamos
        split="val",
    )

    # ------------------ Modelos ------------------
    # CVAE pequeña (Stage-1) que provee latentes z al generador condicional (Stage-2)
    cvae = CVAE(
        z_dim=32,
        cond_dim=4,
        seq_len=seq_len,
        ev_embed=16,
        cond_embed=4,
        enc_hid=16,
        dec_hid=16,
    ).to(device)

    # Generador y Crítico de la cGAN condicional (dimensiones acordes a seq_len)
    gen = Generator(
        z_dim=32,
        cond_dim=4,
        seq_len=seq_len,
        hidden_dim=32
    ).to(device)

    # Crítico/Discriminador
    disc = Critic(
        cond_dim=4,
        seq_len=seq_len,
        hidden_dim=32
    ).to(device)

    # ------------------ Cargar CVAE preentrenada ------------------
    ckpt_cvae = Path("checkpoints/cvae_pretrained_best.pth") # ruta del checkpoint
    assert ckpt_cvae.exists(), "Falta checkpoints/cvae_pretrained_best.pth (ejecuta train_cvae.py primero)." # verificar existencia
    cvae.load_state_dict(torch.load(ckpt_cvae, map_location=device)) # cargar pesos
    for p in cvae.parameters():
        p.requires_grad_(False)  # congela CVAE (no se entrena en Stage-2)

    # ------------------ Optimizadores ------------------
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    # ------------------ Configuración WGAN-GP ------------------
    config = {
        "lambda_gp": 10, # peso del grad penalty
        "critic_iters": 5 # veces que se actualiza D por cada actualización de G
    }

    # ------------------ Loop de entrenamiento ------------------
    best_val_g = float("inf")  # criterio: minimizar loss_G en validación
    Path("checkpoints").mkdir(exist_ok=True) # carpeta de checkpoints si no existe

    total_epochs = 20
    for epoch in range(total_epochs):
        logging.info(f"=== Epoch {epoch+1}/{total_epochs} (Stage-2) ===")

        # ---- entrenamiento de la época ----
        train_stats = train_cgan_epoch(gen, disc, cvae, dataloader, opt_g, opt_c, device, config) # entrenar época
        if train_stats is None:
            logging.warning("Entrenamiento sin pasos válidos.") 
            continue
        logging.info(
            f"[Train] D={train_stats['loss_D']:.3f} | G={train_stats['loss_G']:.3f} | "
            f"D_real={train_stats['D_real']:.3f} D_fake={train_stats['D_fake']:.3f} | "
            f"W={train_stats['W']:.3f} | GP={train_stats['GP']:.3f} | steps={train_stats['steps']}"
        )

        # ---- validación de la época ----
        val_stats = validate_cgan_epoch(gen, disc, cvae, val_loader, device, config) # validar época
        if val_stats is None:
            logging.warning("Validación sin pasos válidos.")
            continue
        logging.info(
            f"[Val]   D={val_stats['loss_D']:.3f} | G={val_stats['loss_G']:.3f} | "
            f"D_real={val_stats['D_real']:.3f} D_fake={val_stats['D_fake']:.3f} | "
            f"W={val_stats['W']:.3f}"
        )

        # Guardar mejores pesos cuando la loss_G de validación mejora (más negativa = mejor en WGAN)
        if val_stats["loss_G"] < best_val_g:
            best_val_g = val_stats["loss_G"] # nueva mejor pérdida
            torch.save(gen.state_dict(), "checkpoints/generator_pretrained_best.pth") # guardar G
            torch.save(disc.state_dict(), "checkpoints/critic_pretrained_best.pth") # guardar D
            logging.info(f"Nuevo mejor val loss_G={best_val_g:.3f} -> checkpoints/_best.pth") # guardar checkpoint

    # Guardamos por si acaso la última
    torch.save(gen.state_dict(),  "checkpoints/generator_pretrained_last.pth") # guardar G
    torch.save(disc.state_dict(), "checkpoints/critic_pretrained_last.pth") # guardar D
    logging.info("C-GAN preentrenada (acoplada a CVAE) guardada en checkpoints/") # guardar checkpoint final

if __name__ == "__main__":
    main() 