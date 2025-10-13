# data_pipeline.py
# ============================================================
# Dataset MIDI para CVAE + cGAN condicionado por género
# - SOLO usa etiquetas desde CSV (path, genre_id)
# - Extrae piano-roll y eventos; trocea por seq_len; soporta cache
# ============================================================

import os
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import pickle

import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# Utilidades de conversión
# -------------------------------
def midi_to_piano_roll(midi_path: Path, fs: int = 16) -> Optional[np.ndarray]:
    """
    Convierte un archivo MIDI en una matriz piano-roll binaria (128 x T).
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        piano_roll = midi_data.get_piano_roll(fs=fs)
        return (piano_roll > 0).astype(np.float32)
    except Exception as e:
        logging.warning(f"No se pudo procesar {midi_path}: {e}")
        return None

def midi_to_event_encoding(midi_path: Path) -> Optional[np.ndarray]:
    """
    Extrae eventos (timestamp, pitch, velocity) de un archivo MIDI.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        events = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                events.append((note.start, note.pitch, note.velocity))
        events.sort(key=lambda x: x[0])
        return np.array(events, dtype=np.float32)
    except Exception as e:
        logging.warning(f"No se pudo procesar eventos de {midi_path}: {e}")
        return None

# -------------------------------
# Normalización de rutas (CSV)
# -------------------------------
def normalize_csv_path(path_str: str, root_dir: Path) -> Path:
    """
    Normaliza la ruta venida del CSV y la rehace relativa a root_dir.
    - Convierte separadores \ ↔ /.
    - Recorta './' inicial.
    - Si es absoluta, intenta relativizarla a root_dir; si no se puede, la usa tal cual.
    """
    s = (path_str or '').strip().replace('\\', '/')
    s = s.lstrip('./')
    p = Path(s)
    if p.is_absolute():
        try:
            rel = p.relative_to(root_dir)
            return root_dir / rel
        except Exception:
            return p
    return root_dir / p

# ============================================================
# Dataset
# ============================================================
class MIDIDataset(Dataset):
    """
    Dataset para CVAE + cGAN:
      - Construye secuencias (piano-roll) y eventos por ventana de seq_len.
      - Etiqueta por género (0..3) desde CSV.
      - Cache opcional.
    """

    def __init__(
        self,
        midi_root: str,                    # raíz donde viven los .mid
        seq_len: int,
        fs: int = 16,
        cache: bool = True,
        label_mode: str = "csv",           # "csv" 
        csv_path: Optional[str] = None,    # requerido si label_mode="csv"
        csv_path_column: str = "path",
        csv_label_column: str = "genre_id",
    ):
        """
        Args:
            midi_root: carpeta raíz del dataset MIDI ('data/Lakh_MIDI_Dataset_Clean')
            seq_len: longitud temporal por muestra
            fs: frames por segundo para piano-roll
            cache: guardar/leer cache .npz
            label_mode:
                - "csv": usa CSV para (path -> genre_id)
                - "synthetic": adjunta géneros aleatorios (debug)
            csv_path: ruta al CSV (si label_mode="csv")
            csv_path_column: nombre de columna con la ruta del MIDI
            csv_label_column: nombre de columna con el genre_id (0..3)
        """
        self.midi_root = Path(midi_root)
        self.seq_len = seq_len
        self.fs = fs
        self.cache = cache
        self.label_mode = label_mode
        self.csv_path = Path(csv_path) if csv_path else None
        self.csv_path_column = csv_path_column
        self.csv_label_column = csv_label_column

        if not self.midi_root.is_dir():
            raise FileNotFoundError(f"Raíz MIDI no encontrada: {midi_root}")

        if self.label_mode == "csv" and (self.csv_path is None or not self.csv_path.is_file()):
            raise FileNotFoundError("label_mode='csv' requiere csv_path válido con columnas 'path' y 'genre_id'.")

        self.sequences: List[np.ndarray] = []
        self.event_encodings: List[np.ndarray] = []
        self.labels: List[int] = []

        self._prepare_data()

    # -------------------------------
    def _load_csv_index(self) -> List[Tuple[Path, int]]:
        """
        Lee el CSV y retorna lista de (ruta_normalizada, genre_id).
        """
        df = pd.read_csv(self.csv_path)
        if self.csv_path_column not in df.columns or self.csv_label_column not in df.columns:
            raise ValueError(f"CSV debe incluir columnas '{self.csv_path_column}' y '{self.csv_label_column}'.")

        pairs: List[Tuple[Path, int]] = []
        missing = 0
        for _, row in df.iterrows():
            rel = str(row[self.csv_path_column])
            gid = int(row[self.csv_label_column])
            midi_path = normalize_csv_path(rel, self.midi_root)
            if not midi_path.is_file():
                missing += 1
                continue
            # Acepta .mid o .midi
            if midi_path.suffix.lower() not in (".mid", ".midi"):
                continue
            pairs.append((midi_path, gid))

        if missing > 0:
            logging.warning(f"[CSV] {missing} rutas del CSV no se encontraron bajo {self.midi_root}. Se omiten.")

        logging.info(f"[CSV] Cargados {len(pairs)} archivos validados desde {self.csv_path}.")
        return pairs

    # -------------------------------
    def _iter_files_and_labels(self) -> List[Tuple[Path, int]]:
        """
        Retorna la lista de (ruta_midi, genre_id) según label_mode.
        """
        if self.label_mode == "csv":
            return self._load_csv_index()

        # synthetic (debug): escanea raíz y asigna etiqueta aleatoria
        midi_paths = list(self.midi_root.rglob("*.mid")) + list(self.midi_root.rglob("*.midi"))
        pairs = [(p, random.randint(0, 3)) for p in midi_paths]
        logging.info(f"[SYNTHETIC] Etiquetas aleatorias para {len(pairs)} archivos.")
        return pairs

    # -------------------------------
    def _prepare_data(self):
        logging.info(f"Preparando datos: seq_len={self.seq_len}, fs={self.fs}, label_mode={self.label_mode}")

        # cache separado por modo de labels
        cache_suffix = "csv" if self.label_mode == "csv" else self.label_mode
        cache_path = self.midi_root / f"cache_{cache_suffix}_seq{self.seq_len}.pkl"

        if self.cache and cache_path.exists():
            logging.info(f"Cargando cache desde {cache_path}")
            with open(cache_path, 'rb') as handle:
                data = pickle.load(handle)
            self.sequences = list(data["sequences"])
            self.event_encodings = list(data["event_encodings"])
            self.labels = list(data["labels"])
            logging.info(f"Dataset cacheado: {len(self.sequences)} secuencias.")
            return

        file_label_pairs = self._iter_files_and_labels()
        if not file_label_pairs:
            raise ValueError("No hay archivos MIDI etiquetados para procesar.")

        for midi_path, genre_id in tqdm(file_label_pairs[:100], desc="Procesando archivos MIDI"):
            piano_roll = midi_to_piano_roll(midi_path, self.fs)
            if piano_roll is None or piano_roll.shape[1] < self.seq_len:
                continue

            events = midi_to_event_encoding(midi_path)

            T = piano_roll.shape[1]
            # troceo no solapado; si quieres overlap usa step < seq_len
            for i in range(0, T - self.seq_len + 1, self.seq_len):
                seq = piano_roll[:, i:i + self.seq_len]
                self.sequences.append(seq)

                if events is not None:
                    t0, t1 = i / self.fs, (i + self.seq_len) / self.fs
                    mask = (events[:, 0] >= t0) & (events[:, 0] < t1)
                    seg_events = events[mask]
                    self.event_encodings.append(seg_events)
                else:
                    self.event_encodings.append(np.zeros((0, 3), dtype=np.float32))

                self.labels.append(int(genre_id))

        if not self.sequences:
            raise RuntimeError("No se generaron secuencias. Revisa rutas/CSV y parámetros.")

        if self.cache:
            data_to_cache = {
                'sequences': self.sequences,
                'event_encodings': self.event_encodings,
                'labels': self.labels
            }
            with open(cache_path, 'wb') as handle:
                pickle.dump(data_to_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Cache guardado en {cache_path}")

        logging.info(f"Total de secuencias: {len(self.sequences)}")

    # -------------------------------
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])          # (128, T)
        events = torch.from_numpy(self.event_encodings[idx]) # (N, 3)
        genre_id = int(self.labels[idx])
        cond_vec = torch.tensor([genre_id], dtype=torch.long)
        return {"piano_roll": seq, "events": events, "conditions": cond_vec}