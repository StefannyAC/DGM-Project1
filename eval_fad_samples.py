# eval_fad.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda con asistencia de ChatGPT
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Evalúa FAD (Frechet Audio Distance) por género y global en muestras de latentes aleatorias
# a partir de checkpoints del modelo híbrido (CVAE + Generator).
# - Usa el split de TEST del data_pipeline
# - Selecciona automáticamente "best" y si no existe cae a "last"
# - Crea muestras reales (ground-truth) y falsas (CVAE+G) -> WAV
# - Extrae embeddings con VGGish y calcula FAD
# ============================================================

import os # para manejo de rutas
os.environ.setdefault("SDL_AUDIODRIVER", "dummy") # desactiva salida de audio real
os.environ.setdefault("FLUIDSYNTH_AUDIO_DRIVER", "wasapi") # define driver de audio Fluidsynth
import math # funciones matemáticas
import json # manejo de JSON
import time # medir tiempos
import shutil # operaciones con archivos y carpetas
import logging # logging de información
from pathlib import Path # manejo de rutas
from collections import defaultdict, Counter # conteo y agrupación de elementos

import numpy as np # manejo de arrays
import pandas as pd # manejo de dataframes
import soundfile as sf # lectura y escritura de archivos de audio

import torch # Para Tensores
import torch.nn.functional as F # Para funciones de activación y pérdidas
from tqdm import tqdm # barra de progreso

from data_pipeline import get_split_dataloader # Para cargar datos
from cvae_seq2seq import CVAE # Modelo CVAE
from cgan import Generator # Generador del C-GAN
import pretty_midi # procesamiento de MIDI 

from transformers import AutoFeatureExtractor, ASTModel # modelo AST preentrenado
import librosa # análisis y carga de audio
from scipy.linalg import sqrtm # raíz matricial (FID, métricas)

# Configuración básica de logging: timestamp + solo mensaje
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ============================
# Config
# ============================
CONFIG = {
    # audio render
    "sample_rate": 16000,   # AST requiere 16 kHz
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
    "stages_T": [32], # solo queremos el final

    # carpetas salida
    "outdir": "eval_fad_hybrid_samples_Transformer",
    "results_csv": "eval_fad_hybrid_samples_Transformer/fad_results.csv",

    # pesos del transformer
    "ast_model_id": "MIT/ast-finetuned-audioset-10-10-0.4593",
}


# ============================
# Utilidades: Piano-roll <-> MIDI <-> WAV
# ============================
def pianoroll_to_pretty_midi(X_bin, fs=16, program=0, velocity=80, min_dur_frames=1):
    """
    Función que convierte una matriz binaria de piano-roll en un objeto PrettyMIDI.

    Args:
        X_bin: Matriz binaria de tamaño (128, T). Los valores son {0,1}.
        fs: Frecuencia temporal (frames por segundo) del piano-roll.
        program: Número del instrumento MIDI (0 = piano acústico).
        velocity: Intensidad (0-127) de las notas MIDI generadas.
        min_dur_frames: Duración mínima (en frames) para considerar una nota válida.

    Returns:
        pretty_midi.PrettyMIDI: Objeto PrettyMIDI que contiene el instrumento y sus notas.
    """
    pm = pretty_midi.PrettyMIDI() # crea objeto MIDI vacío
    inst = pretty_midi.Instrument(program=program) # crea instrumento con el número de programa indicado
    T = X_bin.shape[1] # número total de frames temporales

    for pitch in range(128): # recorre todos los tonos MIDI posibles
        pr = X_bin[pitch] # obtiene la fila correspondiente a este tono
        on = None # marcador de inicio de nota (None = no activa)
        for t in range(T): # recorre el tiempo frame a frame
            if pr[t] == 1 and on is None: # si empieza una nota (paso 0->1)
                on = t # guarda el frame de inicio
            if (on is not None) and (pr[t] == 0): # si termina una nota (paso 1->0)
                dur = t - on # calcula duración en frames
                if dur >= min_dur_frames: # descarta notas demasiado cortas
                    start = on / fs # tiempo inicial en segundos
                    end   = t / fs # tiempo final en segundos
                    inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)) # intensidad, tono, inicio, fin 
                on = None  # resetea marcador de inicio
        if on is not None: # si queda una nota abierta hasta el final
            dur = T - on # calcula duración restante
            if dur >= min_dur_frames: # verifica duración mínima
                start = on / fs # inicio en segundos
                end   = T / fs # fin en segundos (último frame)
                inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))

    pm.instruments.append(inst) # añade el instrumento al objeto MIDI
    return pm # devuelve el objeto PrettyMIDI final

@contextmanager
def suppress_stderr():
    """
    Contexto para suprimir temporalmente los mensajes de error (stderr).
    """
    devnull = open(os.devnull, 'w') # abre /dev/null para desechar salida
    old_stderr_fd = os.dup(2)  # duplica descriptor de stderr (2)
    try:
        os.dup2(devnull.fileno(), 2) # redirige stderr a /dev/null
        yield # ejecuta el bloque dentro del contexto
    finally:
        os.dup2(old_stderr_fd, 2) # restaura stderr original
        os.close(old_stderr_fd) # cierra descriptor temporal
        devnull.close() # cierra /dev/null

def render_midi_to_wav(pm: pretty_midi.PrettyMIDI, wav_path: Path, sr: int = 16000):
    """
    Renderiza un objeto PrettyMIDI a un archivo WAV.

    Args:
        pm: Objeto MIDI a convertir.
        wav_path: Ruta donde se guardará el archivo WAV.
        sr: Frecuencia de muestreo del audio de salida (por defecto 16 kHz).

    Returns:
        None (genera archivo WAV en disco).
    """
    audio = pm.fluidsynth(fs=sr) # sintetiza audio a partir del MIDI usando fluidsynth
    if audio.ndim == 2: # si es estéreo
        audio = audio.mean(axis=1) # convierte a mono promediando canales
    sf.write(str(wav_path), audio, sr) # guarda el audio en disco

def load_ast_model(model_id, device):
    """
    Carga el modelo AST (Audio Spectrogram Transformer) desde Hugging Face.

    Args:
        model_id: Nombre o ruta del modelo en Hugging Face.
        device: Dispositivo donde cargar el modelo.

    Returns:
        tuple: (modelo AST, extractor de características).
    """
    logging.info(f"Cargando modelo AST: {model_id}") # registro del modelo cargado
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id) # carga extractor
    model = ASTModel.from_pretrained(model_id).to(device).eval() # carga modelo y mueve a dispositivo
    return model, feature_extractor # retorna ambos

# ============================
# Embeddings y FAD
# ============================
@torch.no_grad()
def extract_ast_embeddings(wav_paths, device, model, feature_extractor):
    """
    Extrae embeddings del modelo AST (Audio Spectrogram Transformer) a partir de archivos WAV.

    Args:
        wav_paths: Lista de rutas a los archivos WAV de entrada.
        device: Dispositivo donde correr el modelo.
        model (ASTModel): Modelo preentrenado de Audio Spectrogram Transformer.
        feature_extractor (AutoFeatureExtractor): Extractor de características de Hugging Face.

    Returns:
        Array de tamaño [N, 768] con un embedding promedio por archivo.

    Detalles:
        - Cada audio se carga, convierte a mono y se remuestrea (si es necesario) a 16 kHz.
        - El modelo AST genera una secuencia de embeddings temporales que se promedian en el eje temporal.
        - Si algún archivo falla, se omite con advertencia en el log.
    """
    model.eval().to(device) # asegura modo evaluación y mueve a dispositivo
    embs = [] # lista para guardar los embeddings
    
    # AST espera una frecuencia de muestreo de 16000
    target_sr = feature_extractor.sampling_rate 

    for wp in tqdm(wav_paths, desc="Embeddings (AST)", leave=False):
        try:
            # Carga la forma de onda del audio
            waveform, sr = sf.read(str(wp))
            
            # Aseguiramos que el audio sea mono
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            # Re-muestrea si es necesario (aunque ya generamos a 16kHz)
            if sr != target_sr:
                # IMPORTANTE: Se necesita 'pip install librosa' para esto
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

            # Pre-procesa el audio para el modelo AST
            # 'return_tensors="pt"' devuelve tensores de PyTorch
            inputs = feature_extractor(waveform, sampling_rate=target_sr, return_tensors="pt")
            
            # Mueve los datos al dispositivo correcto
            input_values = inputs.input_values.to(device)

            # Pasa los datos por el modelo
            outputs = model(input_values)

            # Obtenemos un embedding por clip promediando los "last_hidden_state"
            # La salida de AST es [Batch, Time, EmbeddingDim], promediamos en el tiempo
            embedding = outputs.last_hidden_state.mean(dim=1)
            embs.append(embedding.cpu().numpy()) # guarda embedding en CPU

        except Exception as e:
            logging.warning(f"No se pudo procesar {wp}: {e}. Saltando...") # mensaje si falla un archivo
            continue
    
    # concatena todos los embeddings o devuelve array vacío si no hay válidos
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)

def gaussian_stats(X):
    """
    Calcula la media y covarianza de un conjunto de embeddings.

    Args:
        X: Matriz de tamaño [N, D], donde N son muestras y D la dimensión de los embeddings.

    Returns:
        tuple: (mu, sigma)
            - mu: Vector medio de tamaño [D].
            - sigma: Matriz de covarianza [D, D].
    """
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calcula la Frechet Distance (FID/FAD) entre dos distribuciones gaussianas.

    Args:
        mu1: Media del primer conjunto de embeddings.
        sigma1: Covarianza del primer conjunto.
        mu2: Media del segundo conjunto.
        sigma2: Covarianza del segundo conjunto.
        eps: Pequeño valor para estabilizar la raíz matricial.

    Returns:
        Valor escalar del Frechet Distance.

    Detalles:
        - Mide la similitud entre dos distribuciones gaussianas.
        - Utilizado en métricas como FAD (Frechet Audio Distance).
    """
    diff = mu1 - mu2 # diferencia entre medias
    covmean = sqrtm(sigma1.dot(sigma2) + np.eye(sigma1.shape[0]) * eps) # raíz de producto de covarianzas
    if np.iscomplexobj(covmean): # elimina posibles valores imaginarios
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)  # fórmula FAD
    return abs(float(fid)) # retorna valor escalar absoluto

# ============================
# Helpers de checkpoints
# ============================
def _pick_ckpt(best_path, last_path):
    """
    Selecciona el checkpoint disponible entre los archivos 'best' y 'last'.

    Args:
        best_path: Ruta al checkpoint con mejor desempeño.
        last_path: Ruta al último checkpoint guardado.

    Returns:
        str: Ruta del checkpoint existente o None si no se encuentra ninguno.

    Detalles:
        - Verifica primero si existe el checkpoint "best".
        - Si no, intenta usar el "last".
        - Si ninguno existe, retorna None.
    """
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
    Genera clips WAV por género a partir del dataloader.

    Args:
        dataloader: DataLoader con batches que contienen piano_roll, events y conditions.
        cvae: Modelo CVAE con encoder y método reparameterize.
        gen: Generador que produce piano rolls a partir de (z, y).
        device: Dispositivo ('cuda', 'mps', 'cpu').
        outdir: Carpeta de salida donde guardar los WAVs.
        per_genre: Número máximo de clips por género a recolectar.
        threshold: Umbral binario para convertir activaciones [0,1] a notas activas.
        fs_pr: Frecuencia de muestreo del piano roll.
        min_dur_frames: Duración mínima (en frames) de una nota.
        velocity: Velocidad MIDI asignada a cada nota.
        program: Programa/instrumento MIDI (0 = piano).
        mode: "real" usa los datos ground-truth, "fake" usa muestras generadas.
        min_notes_threshold: Mínimo de notas para aceptar un clip (evita clips vacíos).
        seed: semilla para reproducibilidad del espacio latente generado aleatoriamente.

    Returns:
        dict: Diccionario {genre_id: [rutas_a_wavs]}.
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed) # sincronizamos

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
                z = torch.randn(B, cvae.z_dim, device=device) # construimos una z aleatoria
                Xg = gen(z, y)                       # (B,128,T), [0,1]; usamos el generador para crear una secuencia de ese z
                Xb = (Xg >= threshold).float()
        else:
            Xb = (X >= threshold).float() # si no es fake evalúa el z real que ya viene de la cvae

        Xb_np = Xb.cpu().numpy()
        y_np = y.squeeze(-1).cpu().numpy()

        for i in range(B):
            gid = int(y_np[i])
            if per_genre_count[gid] >= per_genre: # si ya tiene suficientes clips
                continue

            pm = pianoroll_to_pretty_midi( # convierte a objeto MIDI
                Xb_np[i], fs=fs_pr, program=program, velocity=velocity, min_dur_frames=min_dur_frames
            )
            # Si el MIDI no tiene notas, es un clip silencioso y lo ignoramos.
            # Esto evita crear WAVs vacíos.
            # Filtramos clips silenciosos O con muy pocas notas
            num_notes = len(pm.instruments[0].notes) 
            if num_notes < min_notes_threshold: # descarta si hay pocas
                continue
            stem = f"{mode}_g{gid}_{per_genre_count[gid]:06d}" # nombre base del archivo
            wav_path = outdir / f"{stem}.wav" # ruta completa del WAV
            try:
                render_midi_to_wav(pm, wav_path, sr=CONFIG["sample_rate"]) # renderiza a WAV
            except Exception as e:
                logging.warning(f"No se pudo renderizar {stem}: {e}") # log si falla
                continue

            per_genre_paths[gid].append(str(wav_path)) # guarda ruta del WAV
            per_genre_count[gid] += 1 # incrementa conteo del género

        # terminar si ya tenemos todo
        if all(per_genre_count[g] >= per_genre for g in CONFIG["genres"].keys()):
            break

    return per_genre_paths # devuelve las rutas agrupadas por género


# ============================
# Pipeline FAD de un checkpoint
# ============================
def fad_for_checkpoint(T, device, ast_model, feature_extractor):
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
    cvae = CVAE(z_dim=32,
        cond_dim=4,
        seq_len=T,
        ev_embed=16,
        cond_embed=4,
        enc_hid=16,
        dec_hid=16).to(device)
    cvae.load_state_dict(torch.load(cvae_ckpt, map_location=device))
    cvae.eval()

    gen = Generator(z_dim=32, cond_dim=4, seq_len=T).to(device)
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

    # recolectar audios reales y generados
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

    # embeddings y FAD por género
    results = []
    all_real, all_fake = [], []
    for gid, name in CONFIG["genres"].items():
        r_list = real_paths.get(gid, [])
        f_list = fake_paths.get(gid, [])
        if len(r_list) == 0 or len(f_list) == 0:
            logging.warning(f"[T={T}] Género {gid} ({name}) sin suficientes clips.")
            continue

        r_emb = extract_ast_embeddings(r_list, device, ast_model, feature_extractor)
        f_emb = extract_ast_embeddings(f_list, device, ast_model, feature_extractor)

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

    # Sanity: FAD real vs real (split) revisamos qué tan buenas son las reconstrucciones de datos reales
    all_real_sanity = sum((real_paths[g] for g in CONFIG["genres"].keys()), [])
    mid = len(all_real_sanity)//2
    A, B = all_real_sanity[:mid], all_real_sanity[mid:mid*2]
    A_emb = extract_ast_embeddings(A, device, ast_model, feature_extractor)
    B_emb = extract_ast_embeddings(B, device, ast_model, feature_extractor)
    muA, sigA = gaussian_stats(A_emb)
    muB, sigB = gaussian_stats(B_emb)
    fad_rr = frechet_distance(muA, sigA, muB, sigB)
    print(f"\n[Sanity] FAD(real-vs-real) = {fad_rr:.2f}\n")

    # FAD global (macro, juntando todos los emb)
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

    # Inicializa AST UNA sola vez
    ast_model, feature_extractor = load_ast_model(CONFIG["ast_model_id"], device)

    all_rows = []
    for T in CONFIG["stages_T"]:
        logging.info(f"=== Evaluando FAD para T={T} ===")
        try:
            rows = fad_for_checkpoint(T, device, ast_model, feature_extractor) # Aplicamos el FAD usando AST
            all_rows.extend(rows)
        except AssertionError as e:
            logging.warning(str(e))
            continue
        
    # Guardamos 
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