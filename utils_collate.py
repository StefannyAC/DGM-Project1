# utils_collate.py
import torch

def collate_padded(batch):
    """
    batch: lista de dicts con keys: piano_roll (128,T), events (N,3), conditions (1,)
    Devuelve:
      piano_roll: (B,128,T)   (se apila directo)
      events:     (B,N_max,3) (padding con ceros)
      conditions: (B,1)
    """
    # piano-rolls (todos mismo T por seq_len => stack directo)
    X = torch.stack([b["piano_roll"] for b in batch], dim=0)  # (B,128,T)

    # conditions
    y = torch.stack([b["conditions"] for b in batch], dim=0)  # (B,1)

    # events: padding
    evs = [b["events"] for b in batch]
    if len(evs) == 0:
        E = torch.zeros(0, 0, 3)
    else:
        n_max = max(e.shape[0] for e in evs)
        B = len(evs)
        E = torch.zeros(B, n_max, 3, dtype=evs[0].dtype)
        for i, e in enumerate(evs):
            if e.numel() > 0:
                E[i, :e.shape[0]] = e
    return {"piano_roll": X, "events": E, "conditions": y}
