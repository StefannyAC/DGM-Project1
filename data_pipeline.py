# data_pipeline.py

import torch
import numpy as np
import pretty_midi
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
import logging
import random
import json

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ------------------------------------------------------------
# Función: Convertir MIDI a piano-roll binario
# ------------------------------------------------------------
def midi_to_piano_roll(midi_path: Path, fs: int = 16) -> np.ndarray | None:
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


# ------------------------------------------------------------
# Función: Convertir MIDI a codificación basada en eventos
# ------------------------------------------------------------
def midi_to_event_encoding(midi_path: Path) -> np.ndarray | None:
    """
    Extrae eventos (timestamp, pitch, velocity) de un archivo MIDI.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        events = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                events.append((note.start, note.pitch, note.velocity))
        events.sort(key=lambda x: x[0])  # ordenar por tiempo
        return np.array(events, dtype=np.float32)
    except Exception as e:
        logging.warning(f"No se pudo procesar eventos de {midi_path}: {e}")
        return None


# ------------------------------------------------------------
# Clase: MIDIDataset (CVAE + cGAN condicionado por género)
# ------------------------------------------------------------
class MIDIDataset(Dataset):
    """
    Dataset extendido para CVAE + cGAN:
    - Genera piano-roll y codificación por eventos.
    - Condiciona únicamente por género musical.
    - Compatible con Lakh Pianoroll Dataset (LPD-Cleansed) u otros datasets organizados por carpeta de género.
    - Soporta curriculum learning (longitudes variables).
    """

    def __init__(
        self,
        midi_dir: str,
        seq_len: int,
        fs: int = 16,
        cache: bool = True,
        label_mode: str = "folder"  # "folder" o "synthetic"
    ):
        """
        Args:
            midi_dir (str): Carpeta raíz con subcarpetas de géneros (ej. classical/, jazz/, rock/, pop/).
            seq_len (int): Longitud de las secuencias temporales.
            fs (int): Frecuencia de muestreo (time steps por segundo).
            cache (bool): Guardar resultados preprocesados en .npz.
            label_mode (str): "folder" usa el nombre de la carpeta como género; "synthetic" genera aleatorio.
        """
        self.midi_path = Path(midi_dir)
        self.seq_len = seq_len
        self.fs = fs
        self.cache = cache
        self.label_mode = label_mode
        self.sequences = []
        self.event_encodings = []
        self.labels = []

        if not self.midi_path.is_dir():
            raise FileNotFoundError(f"Directorio no encontrado: {midi_dir}")

        # Mapear géneros conocidos a IDs
        self.genre_map = {
            "classical": 0,
            "jazz": 1,
            "rock": 2,
            "pop": 3,
            "electronic": 4
        }

        self._prepare_data()

    # --------------------------------------------------------
    def _prepare_data(self):
        """
        Convierte los archivos MIDI en secuencias de longitud fija,
        y genera representaciones duales (piano-roll y eventos).
        """
        logging.info(f"Preparando datos con secuencia de longitud {self.seq_len}...")

        all_midi_files = list(self.midi_path.glob("**/*.mid")) + list(self.midi_path.glob("**/*.midi"))
        if not all_midi_files:
            raise ValueError(f"No se encontraron archivos MIDI en {self.midi_path}")

        cache_path = self.midi_path / f"cache_seq{self.seq_len}_genre.npz" 
        if self.cache and cache_path.exists():
            logging.info(f"Cargando dataset cacheado desde {cache_path}")
            cache_data = np.load(cache_path, allow_pickle=True)
            self.sequences = list(cache_data["sequences"])
            self.event_encodings = list(cache_data["event_encodings"])
            self.labels = list(cache_data["labels"])
            return

        for midi_file in tqdm(all_midi_files, desc="Procesando archivos MIDI"):
            piano_roll = midi_to_piano_roll(midi_file, self.fs)
            events = midi_to_event_encoding(midi_file)

            if piano_roll is None or piano_roll.shape[1] < 2:
                continue

            genre_id = self._get_genre_label(midi_file)

            # Dividir en fragmentos de seq_len
            total_timesteps = piano_roll.shape[1]
            for i in range(0, total_timesteps - self.seq_len + 1, self.seq_len):
                seq = piano_roll[:, i:i + self.seq_len]
                self.sequences.append(seq)

                # Generar codificación por eventos correspondiente (simplificada)
                if events is not None:
                    mask = (events[:, 0] >= i / self.fs) & (events[:, 0] < (i + self.seq_len) / self.fs)
                    segment_events = events[mask]
                    self.event_encodings.append(segment_events)
                else:
                    self.event_encodings.append(np.zeros((0, 3), dtype=np.float32))

                # Etiqueta de género
                self.labels.append(genre_id)

        if not self.sequences:
            raise RuntimeError("No se pudo generar ninguna secuencia válida.")

        if self.cache:
            np.savez_compressed(
                cache_path,
                sequences=self.sequences,
                event_encodings=self.event_encodings,
                labels=self.labels
            )
            logging.info(f"Dataset cacheado en {cache_path}")

        logging.info(f"Total de secuencias generadas: {len(self.sequences)}")

    # --------------------------------------------------------
    def _get_genre_label(self, midi_file: Path) -> int:
        """
        Obtiene el género según la carpeta (modo 'folder') o genera aleatoriamente (modo 'synthetic').
        """
        if self.label_mode == "folder":
            parent = midi_file.parent.name.lower()
            for genre, gid in self.genre_map.items():
                if genre in parent:
                    return gid
            return random.randint(0, len(self.genre_map) - 1)
        else:
            return random.randint(0, len(self.genre_map) - 1)

    # --------------------------------------------------------
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx])  # (128, T)
        events = torch.from_numpy(self.event_encodings[idx])  # (N, 3)
        genre_id = self.labels[idx]

        cond_vec = torch.tensor([genre_id], dtype=torch.long)  # solo género
        return {
            "piano_roll": seq,
            "events": events,
            "conditions": cond_vec
        }
