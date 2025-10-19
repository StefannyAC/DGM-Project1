# utils_collate.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda con asistencia de ChatGPT
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
import torch

def collate_padded(batch):
    """
    Función para agrupar (collate) muestras heterogéneas del dataset en un batch,
    aplicando padding sobre la dimensión de eventos (N) y apilando directamente
    las matrices piano-roll (todas comparten 'seq_len' -> mismo T).

    Args:
        batch: Lista de muestras, cada una con claves:
            - "piano_roll": torch.Tensor de forma (128, T) y valores {0,1}.
            - "events":     torch.Tensor de forma (N, 3) con eventos (t, pitch, vel), N variable.
            - "conditions": torch.LongTensor de forma (1,) con el 'genre_id'.

    Returns:
        dict: Diccionario con tensores batcheados:
            - "piano_roll": torch.Tensor (B, 128, T)  - apilado directo (sin padding).
            - "events":     torch.Tensor (B, N_max, 3) - con padding de ceros cuando N < N_max.
            - "conditions": torch.LongTensor (B, 1).
    """
    # piano-rolls (todos mismo T por seq_len => stack directo)
    X = torch.stack([b["piano_roll"] for b in batch], dim=0)  # (B,128,T)

    # conditions
    y = torch.stack([b["conditions"] for b in batch], dim=0)  # (B,1)

    # events: padding
    evs = [b["events"] for b in batch]
    if len(evs) == 0:
        E = torch.zeros(0, 0, 3) # Caso extremo: batch vacío (no debería ocurrir con DataLoader), crea tensor vacío coherente.
    else:
        n_max = max(e.shape[0] for e in evs)
        B = len(evs)
        E = torch.zeros(B, n_max, 3, dtype=evs[0].dtype)
        for i, e in enumerate(evs):
            if e.numel() > 0:
                E[i, :e.shape[0]] = e # Copia cada secuencia de eventos hasta su longitud N_i; el resto queda en 0 (padding).
    return {"piano_roll": X, "events": E, "conditions": y}
