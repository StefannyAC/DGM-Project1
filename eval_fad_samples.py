# eval_fad.py
# ============================================================
# Evalúa FAD (Frechet Audio Distance) por género y global
# a partir de checkpoints del modelo híbrido (CVAE + Generator).
# - Usa el split de TEST del data_pipeline
# - Selecciona automáticamente "best" y si no existe cae a "last"
# - Crea muestras reales (ground-truth) y falsas (CVAE+G) -> WAV
# - Extrae embeddings con VGGish y calcula FAD
# ============================================================

import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("FLUIDSYNTH_AUDIO_DRIVER", "wasapi")
import math
import json
import time
import shutil
import logging
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_pipeline import get_split_dataloader
from cvae_seq2seq import CVAE
from cgan import Generator
import pretty_midi

# --- VGGish ---
# pip install torchvggish
from torchvggish import vggish, vggish_input


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============================
# Config
# ============================
CONFIG = {
    # audio render
    "sample_rate": 16000,   # VGGish requiere 16 kHz
    "fs_pr": 16,            # frames/s del piano-roll (de data_pipeline)

    # evaluación
    "genres": {0: "classical", 1: "jazz", 2: "rock", 3: "pop"},
    "per_genre_eval_samples": 200,   # clips por género (reales y generados)
    "threshold": 0.5,                # umbral piano-roll -> notas on/off
    "note_min_dur_frames": 2,        # duración mínima en frames (fs_pr)
    "velocity": 80,
    "program": 0,                    # piano acústico GM

    # curriculum a evaluar
    # "stages_T": [32, 64, 128], # si queremos revisar la evolución completa
    "stages_T": [128], # solo queremos el final

    # carpetas salida
    "outdir": "eval_fad_hybrid_samples",
    "results_csv": "eval_fad_hybrid_samples/fad_results.csv",

    # pesos VGGish (None = aleatorios)
    "vggish_ckpt": "checkpoints/vggish-10086976.pth",
}


# ============================
# Utilidades: Piano-roll <-> MIDI <-> WAV
# ============================
def pianoroll_to_pretty_midi(X_bin, fs=16, program=0, velocity=80, min_dur_frames=1):
    """
    X_bin: (128, T) binaria {0,1}
    Devuelve pretty_midi.PrettyMIDI.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    T = X_bin.shape[1]

    for pitch in range(128):
        pr = X_bin[pitch]
        on = None
        for t in range(T):
            if pr[t] == 1 and on is None:
                on = t
            if (on is not None) and (pr[t] == 0):
                dur = t - on
                if dur >= min_dur_frames:
                    start = on / fs
                    end   = t / fs
                    inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
                on = None
        if on is not None:
            dur = T - on
            if dur >= min_dur_frames:
                start = on / fs
                end   = T / fs
                inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))

    pm.instruments.append(inst)
    return pm

from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    devnull = open(os.devnull, 'w')
    old_stderr_fd = os.dup(2)  # 2 = stderr
    try:
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(old_stderr_fd, 2)
        os.close(old_stderr_fd)
        devnull.close()

def render_midi_to_wav(pm: pretty_midi.PrettyMIDI, wav_path: Path, sr: int = 16000):
    """
    Render sencillo vía pretty_midi.fluidsynth (usa Fluidsynth instalado en el sistema).
    Salida mono 16 kHz.
    """
    audio = pm.fluidsynth(fs=sr)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    sf.write(str(wav_path), audio, sr)

def load_vggish(device, ckpt_path=None, use_postprocess=True):
    """
    Carga el modelo VGGish y se asegura de que TODOS sus componentes
    (incluyendo los tensores PCA no registrados) estén en el dispositivo correcto.
    """
    model = vggish()
    model.postprocess = use_postprocess  # True para usar PCA

    if ckpt_path:
        if Path(ckpt_path).is_file():
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state)
        else:
            logging.warning(f"No se encontró el checkpoint VGGish en: {ckpt_path}. "
                            "Se usarán pesos aleatorios (métricas poco fiables).")
    else:
        logging.warning("Sin checkpoint de VGGish: usando pesos aleatorios.")

    # Mueve el modelo completo al dispositivo
    model.to(device)

    # --- INICIO DE LA CORRECCIÓN CLAVE ---
    # Forzamos manualmente el movimiento de los tensores PCA al dispositivo correcto,
    # ya que no están registrados correctamente en el módulo y .to(device) los ignora.
    # --- MUY IMPORTANTE: mover PCA al mismo device ---
    if getattr(model, "pproc", None) is not None:
        if hasattr(model.pproc, "_pca_matrix"):
            model.pproc._pca_matrix = model.pproc._pca_matrix.to(device)
        if hasattr(model.pproc, "_pca_means"):
            model.pproc._pca_means = model.pproc._pca_means.to(device)
    # --- FIN DE LA CORRECCIÓN CLAVE ---

    return model

# ============================
# Embeddings y FAD
# ============================
@torch.no_grad()
def extract_vggish_embeddings(wav_paths, device, model):
    """
    Retorna np.ndarray [N, 128] con embeddings VGGish.
    Soporta que vggish_input.wavfile_to_examples devuelva np.ndarray o torch.Tensor
    con diferentes dimensiones.
    """
    model.eval().to(device)
    embs = []

    for wp in tqdm(wav_paths, desc="Embeddings (VGGish)", leave=False):
        ex = vggish_input.wavfile_to_examples(str(wp))  # Puede ser [N,96,64], [N,1,96,64], etc.

        # --- INICIO DE LA CORRECCIÓN ---

        # 1. Convertir a Tensor de PyTorch si es necesario
        if not isinstance(ex, torch.Tensor):
            ex = torch.from_numpy(np.asarray(ex, dtype=np.float32))

        # 2. Asegurar que el tensor tenga 4 dimensiones [B, C, H, W]
        if ex.ndim == 2:  # Caso: [H, W] -> [96, 64]
            ex = ex.unsqueeze(0).unsqueeze(0)  # -> [1, 1, 96, 64]
        elif ex.ndim == 3:  # Caso: [B, H, W] -> [N, 96, 64]
            ex = ex.unsqueeze(1)  # -> [N, 1, 96, 64]
        
        # Si ex.ndim ya es 4, asumimos que tiene la forma correcta [N, 1, 96, 64] y no hacemos nada.
        # Si tiene 5 dimensiones o más, es un caso inesperado, pero esta lógica evita añadir más.

        x = ex.to(device).float()

        # --- FIN DE LA CORRECCIÓN ---

        if x.shape[0] == 0:
            # audio demasiado corto → clip silencioso
            # Usamos el mismo device y dtype para evitar errores de compatibilidad
            x = torch.zeros(1, 1, 96, 64, device=device, dtype=torch.float32)

        z = model(x)                          # [N, 128]
        z = z.mean(dim=0, keepdim=True)     # promedio por clip
        embs.append(z.cpu().numpy())

    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 128), dtype=np.float32)


def gaussian_stats(X):
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    FID/FAD clásico entre Gaussinas N(mu1, sigma1) y N(mu2, sigma2).
    """
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2) + np.eye(sigma1.shape[0]) * eps)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return abs(float(fid))


# ============================
# Helpers de checkpoints
# ============================
def _pick_ckpt(best_path, last_path):
    if Path(best_path).is_file():
        return best_path
    if Path(last_path).is_file():
        return last_path
    return None


# ============================
# Recolección de audio
# ============================
def collect_audio_from_loader(
    dataloader, cvae, gen, device, outdir, per_genre=200, threshold=0.5,
    fs_pr=16, min_dur_frames=1, velocity=80, program=0, mode="real", min_notes_threshold=5,seed=None
):
    """
    Genera WAVs por género desde el dataloader.
    - mode="real": usa X (ground-truth) para MIDI->WAV.
    - mode="fake": z = encoder(X,E,y)-> G(z,y); umbraliza y sintetiza.
    Devuelve: dict {genre_id: [wav_paths...]}
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_genre_count = Counter()
    per_genre_paths = defaultdict(list)

    for batch in tqdm(dataloader, desc=f"Recolectando audio ({mode})", leave=False):
        X = batch["piano_roll"].to(device).float()   # (B,128,T)
        E = batch["events"].to(device).float()
        y = batch["conditions"].to(device)           # (B,1)
        B, C, T = X.shape

        if mode == "fake":
            with torch.no_grad():
                z = torch.randn(B, cvae.z_dim, device=device)
                Xg = gen(z, y)                       # (B,128,T), [0,1]
                Xb = (Xg >= threshold).float()
        else:
            Xb = (X >= threshold).float()

        Xb_np = Xb.cpu().numpy()
        y_np = y.squeeze(-1).cpu().numpy()

        for i in range(B):
            gid = int(y_np[i])
            if per_genre_count[gid] >= per_genre:
                continue

            pm = pianoroll_to_pretty_midi(
                Xb_np[i], fs=fs_pr, program=program, velocity=velocity, min_dur_frames=min_dur_frames
            )
            # --- INICIO DE LA CORRECCIÓN CLAVE ---
            # Si el MIDI no tiene notas, es un clip silencioso y lo ignoramos.
            # Esto evita crear WAVs vacíos que hacen fallar a VGGish.
            # Filtramos clips silenciosos O con muy pocas notas
            num_notes = len(pm.instruments[0].notes)
            if num_notes < min_notes_threshold:
                # logging.debug(f"Clip con {num_notes} notas (<{min_notes_threshold}). Saltando...")
                continue
            # --- FIN DE LA CORRECCIÓN MEJORADA ---
            stem = f"{mode}_g{gid}_{per_genre_count[gid]:06d}"
            wav_path = outdir / f"{stem}.wav"
            try:
                render_midi_to_wav(pm, wav_path, sr=CONFIG["sample_rate"])
            except Exception as e:
                logging.warning(f"No se pudo renderizar {stem}: {e}")
                continue

            per_genre_paths[gid].append(str(wav_path))
            per_genre_count[gid] += 1

        # terminar si ya tenemos todo
        if all(per_genre_count[g] >= per_genre for g in CONFIG["genres"].keys()):
            break

    return per_genre_paths


# ============================
# Pipeline FAD de un checkpoint
# ============================
def fad_for_checkpoint(T, device, vgg_model):
    """
    Calcula FAD (global y por género) para T específico del currículo.
    Devuelve list[dict] con métricas.
    """
    # elige checkpoints best -> last
    best_cvae = f"checkpoints/hybrid_cvae_best_T{T}.pth"
    last_cvae = f"checkpoints/hybrid_cvae_T{T}_last.pth"
    best_gen  = f"checkpoints/hybrid_G_best_T{T}.pth"
    last_gen  = f"checkpoints/hybrid_G_T{T}_last.pth"

    cvae_ckpt = _pick_ckpt(best_cvae, last_cvae)
    gen_ckpt  = _pick_ckpt(best_gen,  last_gen)
    assert cvae_ckpt and gen_ckpt, f"Faltan checkpoints para T={T}"

    # loader TEST para T
    loader = get_split_dataloader(
        seq_len=T,
        batch_size=64,
        num_workers=0,
        use_balanced_sampler=False,   # para no distorsionar stats
        split="test"
    )

    # modelos
    cvae = CVAE(z_dim=128, cond_dim=4, seq_len=T).to(device)
    cvae.load_state_dict(torch.load(cvae_ckpt, map_location=device))
    cvae.eval()

    gen = Generator(z_dim=128, cond_dim=4, seq_len=T).to(device)
    gen.load_state_dict(torch.load(gen_ckpt, map_location=device))
    gen.eval()

    # carpetas de audio (borra y recrea)
    stage_out = Path(CONFIG["outdir"]) / f"T{T}"
    real_dir  = stage_out / "real"
    fake_dir  = stage_out / "fake"
    if stage_out.exists():
        shutil.rmtree(stage_out)
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # 1) recolectar audios reales y generados
    real_paths = collect_audio_from_loader(
        loader, cvae, gen, device, real_dir,
        per_genre=CONFIG["per_genre_eval_samples"],
        threshold=CONFIG["threshold"],
        fs_pr=CONFIG["fs_pr"],
        min_dur_frames=CONFIG["note_min_dur_frames"],
        velocity=CONFIG["velocity"],
        program=CONFIG["program"],
        mode="real",
        min_notes_threshold=5  # Evita clips casi silenciosos
    )
    fake_paths = collect_audio_from_loader(
        loader, cvae, gen, device, fake_dir,
        per_genre=CONFIG["per_genre_eval_samples"],
        threshold=CONFIG["threshold"],
        fs_pr=CONFIG["fs_pr"],
        min_dur_frames=CONFIG["note_min_dur_frames"],
        velocity=CONFIG["velocity"],
        program=CONFIG["program"],
        mode="fake",
        min_notes_threshold=1, # Acepta clips con al menos 1 nota
        seed=123  # para reproducibilidad en fake
    )

    # 2) embeddings y FAD por género
    results = []
    all_real, all_fake = [], []
    for gid, name in CONFIG["genres"].items():
        r_list = real_paths.get(gid, [])
        f_list = fake_paths.get(gid, [])
        if len(r_list) == 0 or len(f_list) == 0:
            logging.warning(f"[T={T}] Género {gid} ({name}) sin suficientes clips.")
            continue

        r_emb = extract_vggish_embeddings(r_list, device, vgg_model)
        f_emb = extract_vggish_embeddings(f_list, device, vgg_model)

        mu_r, sig_r = gaussian_stats(r_emb)
        mu_f, sig_f = gaussian_stats(f_emb)
        fad = frechet_distance(mu_r, sig_r, mu_f, sig_f)

        results.append({
            "T": T, "genre_id": gid, "genre_name": name,
            "n_real": len(r_list), "n_fake": len(f_list),
            "FAD": fad
        })

        all_real.append(r_emb)
        all_fake.append(f_emb)

    # Sanity: FAD real vs real (split)
    all_real_sanity = sum((real_paths[g] for g in CONFIG["genres"].keys()), [])
    mid = len(all_real_sanity)//2
    A, B = all_real_sanity[:mid], all_real_sanity[mid:mid*2]
    A_emb = extract_vggish_embeddings(A, device, vgg_model)
    B_emb = extract_vggish_embeddings(B, device, vgg_model)
    muA, sigA = gaussian_stats(A_emb)
    muB, sigB = gaussian_stats(B_emb)
    fad_rr = frechet_distance(muA, sigA, muB, sigB)
    print(f"\n[Sanity] FAD(real-vs-real) = {fad_rr:.2f}\n")

    # 3) FAD global (macro, juntando todos los emb)
    if len(all_real) and len(all_fake):
        R = np.concatenate(all_real, axis=0)
        Fk = np.concatenate(all_fake, axis=0)
        mu_r, sig_r = gaussian_stats(R)
        mu_f, sig_f = gaussian_stats(Fk)
        fad_global = frechet_distance(mu_r, sig_r, mu_f, sig_f)
        results.append({
            "T": T, "genre_id": -1, "genre_name": "ALL",
            "n_real": R.shape[0], "n_fake": Fk.shape[0],
            "FAD": fad_global
        })
    return results


# ============================
# Main
# ============================
def main():
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    Path(CONFIG["outdir"]).mkdir(parents=True, exist_ok=True)

    # Inicializa VGGish UNA sola vez
    vgg = load_vggish(device, ckpt_path=CONFIG.get("vggish_ckpt"), use_postprocess=False)

    all_rows = []
    for T in CONFIG["stages_T"]:
        logging.info(f"=== Evaluando FAD para T={T} ===")
        try:
            rows = fad_for_checkpoint(T, device, vgg)
            all_rows.extend(rows)
        except AssertionError as e:
            logging.warning(str(e))
            continue

    if all_rows:
        df = pd.DataFrame(all_rows).sort_values(["T", "genre_id"])
        out_csv = CONFIG["results_csv"]
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logging.info(f"Resultados FAD guardados en {out_csv}")
        print(df)
    else:
        logging.warning("No se generaron resultados (¿faltan checkpoints o datos?).")


if __name__ == "__main__":
    main()