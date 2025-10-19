# data_pipeline.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este código define un dataset personalizado para cargar archivos MIDI,
# extraer representaciones piano-roll y de eventos, y preparar los datos
# para entrenar un modelo CVAE + cGAN condicionado por género musical.
# ============================================================
# Dataset MIDI para CVAE + cGAN condicionado por género
# - SOLO usa etiquetas desde CSV (path, genre_id)
# - Extrae piano-roll y eventos; trocea por seq_len; soporta cache; particiona por split
# ============================================================

import os # para manejo de rutas
import json # manejo de JSON
import random # para etiquetas sintéticas
import logging # logging de información
from pathlib import Path # manejo de rutas
from typing import List, Tuple, Optional # anotaciones de tipos
import pickle # para cacheo de datos
from collections import Counter # para conteo de etiquetas

import numpy as np # manejo de arrays
import pandas as pd # manejo de dataframes
import pretty_midi # procesamiento de MIDI
import torch # PyTorch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # dataset y dataloader 
from tqdm import tqdm # barra de progreso

from utils_collate import collate_padded # función de collate personalizada

# Configuración básica de logging: timestamp + solo mensaje
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# Utilidades de conversión
# -------------------------------
def midi_to_piano_roll(midi_path: Path, fs: int = 16) -> Optional[np.ndarray]:
    """
    Función para convertir un archivo MIDI a piano-roll binario (128 x T).

    Args:
        midi_path: Ruta del archivo .mid/.midi a procesar.
        fs: Frecuencia de muestreo en frames por segundo para el piano-roll.

    Returns:
        Optional: Matriz (128, T) con valores {0,1} si se pudo procesar,
            o 'None' si hubo un error/parsing fallido.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path)) # Carga el MIDI con pretty_midi.
        piano_roll = midi_data.get_piano_roll(fs=fs) # Obtiene PR continuo por instrumento y lo suma.
        return (piano_roll > 0).astype(np.uint8) # Binariza: presencia de nota = 1.
    except Exception as e:
        logging.warning(f"No se pudo procesar {midi_path}: {e}") # Log amigable si falla.
        return None # Señaliza fallo devolviendo None.

def midi_to_event_encoding(midi_path: Path) -> Optional[np.ndarray]:
    """
    Función para extraer eventos (timestamp, pitch, velocity) de un MIDI.

    Args:
        midi_path: Ruta del archivo .mid/.midi a procesar.

    Returns:
        Optional: Arreglo de forma (N, 3) con columnas:
            [start_time_seconds, pitch (0-127), velocity (0-127)],
            ordenado por tiempo; 'None' si falla el parsing.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path)) # Carga el MIDI.
        events = [] # Acumula eventos nota a nota
        for instrument in midi_data.instruments: # Itera instrumentos
            for note in instrument.notes: # Itera notas
                events.append((note.start, note.pitch, note.velocity)) # (tiempo, pitch, vel)
        events.sort(key=lambda x: x[0]) # Ordena por tiempo de inicio
        return np.array(events, dtype=np.float32) # Convierte a np.ndarray
    except Exception as e:
        logging.warning(f"No se pudo procesar eventos de {midi_path}: {e}") # Log amigable si falla.
        return None

# -------------------------------
# Normalización de rutas (CSV)
# -------------------------------
def normalize_csv_path(path_str: str, root_dir: Path) -> Path:
    """
    Función para normalizar rutas leídas desde CSV y hacerlas relativas a 'root_dir'.

        - Cambia separadores Windows/Unix.
        - Elimina prefijos './'.
        - Si es absoluta, intenta relativizarla a 'root_dir'; si no se puede, devuelve la absoluta.

    Args:
        path_str: Ruta cruda tal cual viene en el CSV.
        root_dir: Directorio raíz donde deberían residir los MIDIs.

    Returns:
        Path: Ruta normalizada final.
    """
    s = (path_str or '').strip().replace('\\', '/') # normaliza separadores
    s = s.lstrip('./') # elimina prefijos './'
    p = Path(s)  # crea Path
    if p.is_absolute(): # si es absoluta
        try:
            rel = p.relative_to(root_dir) # intenta relativizar
            return root_dir / rel # si se puede, retorna relativa
        except Exception:
            return p # si no, retorna absoluta
    return root_dir / p # si es relativa, la hace relativa a root_dir

# ============================================================
# Dataset
# ============================================================
class MIDIDataset(Dataset):
    """
    Función para construir un Dataset compatible con PyTorch para CVAE + cGAN.

    Genera pares:
        - 'piano_roll': ventanas (128, seq_len) binarizadas,
        - 'events': subarreglo (N, 3) con eventos recortados a la ventana temporal,
        - 'conditions': vector con 'genre_id' (tensor long con shape (1,)).

    Soporta:
        - Etiquetado por CSV (path -> genre_id).
        - Caché opcional de preprocesamiento (pickle) por split y seq_len.

    Args:
        midi_root: Carpeta raíz donde viven los .mid/.midi, por defecto ('data/Lakh_MIDI_Dataset_Clean')
        seq_len: Longitud temporal por muestra (en frames).
        fs: Frames por segundo para el piano-roll.
        cache: Si True, lee/escribe caché procesada (pkl).
        label_mode: "csv" para usar CSV; "synthetic" para etiquetas aleatorias (debug).
        csv_path (Optional): Ruta a CSV si 'label_mode="csv"'.
        csv_path_column: Nombre de columna con la ruta del MIDI.
        csv_label_column: Nombre de columna con la etiqueta (genre_id en 0..3).
        split: Sufijo para separar caches por división ("train"/"val"/"test"); puede ser "".

    Raises:
        FileNotFoundError: si 'midi_root' no existe o si falta el CSV requerido.
        ValueError: si no hay elementos etiquetados o columnas esperadas.
        RuntimeError: si tras el procesamiento no se generaron secuencias.
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
        split: str = "",
    ):
        # Configuración inicial y validaciones
        self.midi_root = Path(midi_root)
        self.seq_len = seq_len
        self.fs = fs
        self.cache = cache
        self.label_mode = label_mode
        self.csv_path = Path(csv_path) if csv_path else None
        self.csv_path_column = csv_path_column
        self.csv_label_column = csv_label_column

        # Validaciones básicas
        if not self.midi_root.is_dir():
            raise FileNotFoundError(f"Raíz MIDI no encontrada: {midi_root}")

        if self.label_mode == "csv" and (self.csv_path is None or not self.csv_path.is_file()):
            raise FileNotFoundError("label_mode='csv' requiere csv_path válido con columnas 'path' y 'genre_id'.")

        # Listas internas para datos procesados
        self.sequences: List[np.ndarray] = []
        self.event_encodings: List[np.ndarray] = []
        self.labels: List[int] = []

        # Split (para separar cachés/CSVs por división)
        self.split = split
        if split != "":
            assert split in str(self.csv_path), "El csv no coincide con el split"
        # Dispara el pipeline de preparación (lectura, parseo, troceo, caché)
        self._prepare_data()

    # -------------------------------
    def _load_csv_index(self) -> List[Tuple[Path, int]]:
        """
        Función para leer el CSV y retorna lista de (ruta_normalizada, genre_id).

        Returns:
            List[Tuple[Path, int]]: Lista de pares (ruta_midi, genre_id) válidos encontrados.
        """
        df = pd.read_csv(self.csv_path) # lee CSV completo
        if self.csv_path_column not in df.columns or self.csv_label_column not in df.columns:
            raise ValueError(f"CSV debe incluir columnas '{self.csv_path_column}' y '{self.csv_label_column}'.") # validación

        pairs: List[Tuple[Path, int]] = [] # acumula pares válidos
        missing = 0 # cuenta archivos faltantes
        for _, row in df.iterrows():
            rel = str(row[self.csv_path_column]) # ruta relativa en CSV
            gid = int(row[self.csv_label_column]) # etiqueta de género
            midi_path = normalize_csv_path(rel, self.midi_root) # normaliza ruta
            if not midi_path.is_file(): # si no existe, cuenta y omite
                missing += 1 # cuenta faltantes
                continue
            # Acepta .mid o .midi
            if midi_path.suffix.lower() not in (".mid", ".midi"):
                continue
            pairs.append((midi_path, gid)) # agrega par válido

        if missing > 0:
            logging.warning(f"[CSV] {missing} rutas del CSV no se encontraron bajo {self.midi_root}. Se omiten.") # log amigable

        logging.info(f"[CSV] Cargados {len(pairs)} archivos validados desde {self.csv_path}.") # log info
        return pairs # retorna pares válidos

    # -------------------------------
    def _iter_files_and_labels(self) -> List[Tuple[Path, int]]:
        """
        Función para obtener la lista de (ruta_midi, genre_id) según label_mode.

        Returns:
            List[Tuple[Path, int]]: Lista de pares (ruta_midi, genre_id) válidos encontrados.
        """
        # Modo CSV: usa CSV para etiquetas
        if self.label_mode == "csv": # usa CSV para etiquetas
            return self._load_csv_index() # carga pares desde CSV

        # Modo synthetic (debug): escanea raíz y asigna etiqueta aleatorias 0...3
        midi_paths = list(self.midi_root.rglob("*.mid")) + list(self.midi_root.rglob("*.midi")) # escanea todos los MIDIs
        pairs = [(p, random.randint(0, 3)) for p in midi_paths] # asigna etiquetas aleatorias
        logging.info(f"[SYNTHETIC] Etiquetas aleatorias para {len(pairs)} archivos.") # log info
        return pairs # retorna pares sintéticos

    # -------------------------------
    def _prepare_data(self):
        """
        Función para preparar el dataset:
            - Carga/crea caché si se solicita.
            - Itera archivos + etiquetas.
            - Convierte a piano-roll, trocea en ventanas 'seq_len'.
            - Recorta eventos por ventana.
            - Empaqueta matrices y guarda listas internas.
        """
        # Intentar cargar caché si existe
        logging.info(f"Preparando datos: seq_len={self.seq_len}, fs={self.fs}, label_mode={self.label_mode}")

        # Sufijo del nombre de caché para diferenciar modos de etiquetado/splits.
        cache_suffix = "csv" if self.label_mode == "csv" else self.label_mode

        # Ruta del archivo de caché diferenciada por split y seq_len
        if self.split == "":
            cache_path = self.midi_root / f"cache_{cache_suffix}_seq{self.seq_len}.pkl"
        else:
            cache_path = self.midi_root / f"cache_{cache_suffix}_seq{self.seq_len}_{self.split}.pkl"

        # Cargar caché si está habilitado y existe
        if self.cache and cache_path.exists():
            logging.info(f"Cargando cache desde {cache_path}")
            with open(cache_path, 'rb') as handle:
                data = pickle.load(handle)
            self.sequences = list(data["sequences"]) 
            self.event_encodings = list(data["event_encodings"]) 
            self.labels = list(data["labels"])
            logging.info(f"Dataset cacheado: {len(self.sequences)} secuencias.")
            return

        # Si no hay caché, procesar desde cero (archivo, etiqueta)
        file_label_pairs = self._iter_files_and_labels()
        if not file_label_pairs:
            raise ValueError("No hay archivos MIDI etiquetados para procesar.")

        for midi_path, genre_id in tqdm(file_label_pairs, desc="Procesando archivos MIDI"):
            piano_roll = midi_to_piano_roll(midi_path, self.fs) # (128, T) binario o None.
            if piano_roll is None or piano_roll.shape[1] < self.seq_len: # Omite si muy corto o falló.
                continue

            events = midi_to_event_encoding(midi_path) # (N, 3) o None.

            T = piano_roll.shape[1] # Largo total en frames.
            # troceo no solapado; si quieres overlap usar step < seq_len
            for i in range(0, T - self.seq_len + 1, self.seq_len):
                seq = piano_roll[:, i:i + self.seq_len] # Submatriz (128, seq_len).
                seq = pack_matrix(seq) # Empaqueta a bytes (2048 bytes).
                self.sequences.append(seq) # Guarda secuencia empaquetada.

                # Recorta eventos que caen dentro de la ventana temporal
                if events is not None:
                    t0, t1 = i / self.fs, (i + self.seq_len) / self.fs # tiempos en segundos
                    mask = (events[:, 0] >= t0) & (events[:, 0] < t1) # máscara temporal
                    seg_events = events[mask] # eventos en ventana
                    self.event_encodings.append(seg_events) # guarda eventos recortados
                else:
                    # Si no hubo eventos, registra arreglo vacío (0x3).
                    self.event_encodings.append(np.zeros((0, 3), dtype=np.float32)) # sin eventos

                self.labels.append(int(genre_id)) # guarda etiqueta

        # Validación final, debe haber secuencias generadas
        if not self.sequences:
            raise RuntimeError("No se generaron secuencias. Revisa rutas/CSV y parámetros.")

        # Guardar caché si está habilitado
        if self.cache:
            data_to_cache = {
                'sequences': self.sequences,
                'event_encodings': self.event_encodings,
                'labels': self.labels
            }
            with open(cache_path, 'wb') as handle:
                pickle.dump(data_to_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Cache guardado en {cache_path}")

        logging.info(f"Total de secuencias: {len(self.sequences)}") # log info final

    # -------------------------------
    def __len__(self):
        """
        Función para retornar la cantidad de muestras en el dataset.

        Returns:
            Número de ventanas (muestras) disponibles.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Función para obtener una muestra indexada del dataset.

        Desempaqueta la secuencia, verifica validez (binaria/finita),
        convierte a tensores y construye el diccionario de salida.

        Args:
            idx: Índice de la muestra (0..len(self)-1).

        Returns:
            Diccionario con claves:
                - "piano_roll": torch.Tensor (128, seq_len) en {0,1}.
                - "events": torch.Tensor (N, 3) de eventos en segundos/pitch/vel.
                - "conditions": torch.LongTensor shape (1,) con genre_id.
        """
        seq_packed = self.sequences[idx] # bytes empaquetados
        seq_unpacked = unpack_matrix(seq_packed, (128, self.seq_len)) # (128, seq_len) 
        # Chequeos de validez: binario y finito
        assert (seq_unpacked >= 0.0).all() and (seq_unpacked <= 1.0).all(), "Not binary"
        assert np.isfinite(seq_unpacked).all(), "Found nan"

        seq = torch.from_numpy(seq_unpacked)          # (128, T)
        events = torch.from_numpy(self.event_encodings[idx]) # (N, 3)
        genre_id = int(self.labels[idx]) # etiqueta entera
        cond_vec = torch.tensor([genre_id], dtype=torch.long) # (1,)

        return {"piano_roll": seq, "events": events, "conditions": cond_vec} # diccionario de salida


def pack_matrix(matrix: np.ndarray) -> bytes:
    """
    Función para empaquetar una matriz binaria 128x128 en una secuencia de bytes usando bit-packing.

    Args:
        matrix: Matriz binaria numpy de forma (128, 128) con valores {0,1}.
    Returns:
        bytes: Secuencia de bytes empaquetada (2048 bytes).
    """
    # Aplana y empaqueta bits en bytes
    packed = np.packbits(matrix.flatten()) # agrupa 8 bits consecutivos del arreglo aplanado en 1 byte.
    return packed.tobytes()  # 128*128 bits / 8 = 2048 bytes


def unpack_matrix(data: bytes, shape: tuple = (128, 128)) -> np.ndarray:
    """
    Función para desempaquetar una secuencia de bytes a una matriz binaria 128x128.

    Args:
        data: Secuencia de bytes empaquetada (2048 bytes).
        shape: Forma de la matriz resultante (por defecto (128, 128)).
    Returns:
        np.ndarray: Matriz binaria numpy de forma (128, 128) con valores {0,1}.
    """
    unpacked = np.unpackbits(np.frombuffer(data, dtype=np.uint8)) # desempaqueta bytes a bits
    return unpacked.astype(np.float32).reshape(shape) # reshape a (128, 128)


def build_loader(midi_root, csv_path, seq_len=128, batch_size=16, num_workers=0,
                 use_balanced_sampler=True, split=""):
    """
    Función para construir un DataLoader de PyTorch para el dataset MIDI,
    con opción de muestreo balanceado por clase (género).

    Args:
        midi_root: Carpeta raíz donde residen los archivos .mid/.midi.
        csv_path: Ruta al CSV con columnas 'path' y 'genre_id'.
        seq_len: Longitud temporal por muestra (en frames) usada por el dataset.
        batch_size: Tamaño de batch.
        num_workers: N° de procesos para cargar datos (0 = hilo principal).
        use_balanced_sampler: Si True, usa 'WeightedRandomSampler' según la inversa de la frecuencia por clase; si False, baraja (shuffle=True).
        split: Etiqueta de división ("train" | "val" | "test" | ""), usada para el nombre del caché y validaciones de CSV.

    Returns:
        DataLoader listo para entrenar/validar/testear.
    """
    # Construye el dataset MIDI
    dataset = MIDIDataset(
        midi_root=midi_root,
        seq_len=seq_len,
        fs=16,
        cache=True,
        label_mode="csv",
        csv_path=csv_path,
        csv_path_column="path",
        csv_label_column="genre_id",
        split=split
    )
    
    if use_balanced_sampler:
        # Cuenta muestras por clase y construye pesos inversos: menos frecuentes -> más peso.
        counts = Counter(dataset.labels)  # ids 0..3
        # inversa de frecuencia por clase
        class_weight = {c: 1.0 / max(n, 1) for c, n in counts.items()}
        sample_w = [class_weight[y] for y in dataset.labels]
        # Construye el sampler balanceado
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        # Construye el DataLoader con el sampler (no se usa shuffle)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_padded,
        )
        logging.info(f"Sampler balanceado activo. Distribución: {dict(counts)}") # log info
    else:
        # DataLoader estándar con shuffle
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_padded,
        )
    return loader # retorna el DataLoader


ROOT = os.path.dirname(__file__) # directorio raíz del proyecto
MIDI_ROOT = os.path.join(ROOT, "dataset/data/Lakh_MIDI_Dataset_Clean") # raíz de MIDIs
CSV_PATH  = os.path.join(ROOT, "dataset/data/lakh_clean_merged_homologado.csv") # CSV con etiquetas

# Validaciones básicas de existencia
if not os.path.exists(MIDI_ROOT):
    raise RuntimeError("MIDIR_ROOT no existe")

if not os.path.exists(CSV_PATH):
    raise RuntimeError("CSV_PATH no existe")

# -------------------------------
def get_csv_path(split: str=""):
    """
    Función para obtener la ruta del CSV según el split.

    Args:
        split: Sufijo del split ("train", "val", "test", o "")
    Returns:
        str: Ruta al CSV correspondiente.
    """
    if split == "":
        return CSV_PATH # sin split, retorna el CSV base

    name, ext = os.path.splitext(CSV_PATH) # separa nombre y extensión
    path = f'{name}_{split}{ext}' # construye ruta con sufijo
    if not os.path.exists(path): # valida existencia
        raise ValueError(f"{path} no existe") # error si no existe
    return path # retorna ruta

# -------------------------------
def get_split_dataset(seq_len: int, split: str=""):
    """
    Función para obtener el dataset MIDI según el split.

    Args:
        seq_len: Longitud temporal por muestra (en frames).
        split: Sufijo del split ("train", "val", "test", o "")
    Returns:
        MIDIDataset: Dataset correspondiente al split.
    """
    return MIDIDataset(
        MIDI_ROOT,
        seq_len,
        csv_path=get_csv_path(split),
        split=split
    )

# -------------------------------
def get_split_dataloader(
    seq_len=128,
    batch_size=16,
    num_workers=0,
    use_balanced_sampler=True,
    split=""
):
    """
    Función para obtener el DataLoader según el split.
    Args:
        seq_len: Longitud temporal por muestra (en frames).
        batch_size: Tamaño de batch.
        num_workers: N° de procesos para cargar datos (0 = hilo principal).
        use_balanced_sampler: Si True, usa muestreo balanceado por clase.
        split: Sufijo del split ("train", "val", "test", o "")
    Returns:
        DataLoader: DataLoader correspondiente al split.
    """
    csv_path = get_csv_path(split)
    print(csv_path)
    return build_loader(
        MIDI_ROOT,
        csv_path,
        seq_len,
        batch_size,
        num_workers,
        use_balanced_sampler,
        split
    )