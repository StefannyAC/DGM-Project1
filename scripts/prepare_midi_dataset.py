# prepare_midi_dataset.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este archivo genera el archivo en caché necesario que guardar las secuencias con las que se entrena, no es necesario ejecutar por separado. 
# ============================================================
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_pipeline import MIDIDataset


def cache_dataset(midi_root: os.PathLike, csv_path: os.PathLike, seq_len: int=128, split: str="") -> None:
    """
    Función para generar/actualizar la caché del MIDIDataset para una combinación '(seq_len, split)'.

    Al instanciar 'MIDIDataset' con 'cache=True', si el archivo de caché no existe, se
    construye y guarda automáticamente; si existe, se omite el procesamiento pesado.

    Args:
        midi_root: Carpeta raíz donde residen los .mid/.midi.
        csv_path: Ruta al CSV del split (p. ej., *_train.csv).
        seq_len: Longitud temporal de ventana (frames) a cachear.
        split: Nombre del split ("train" | "val" | "test" | ""), se usa en el nombre de caché.

    Returns:
        None
    """
    MIDIDataset(
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


def main():
    midi_root = "dataset/data/Lakh_MIDI_Dataset_Clean"
    csv_path  = "dataset/data/lakh_clean_merged_homologado.csv"
    dirname, basename = os.path.split(csv_path)
    name, ext = os.path.splitext(basename)

    seq_lens = [32, 64, 128]
    splits = ['train', 'val', 'test']

    for seq_len in seq_lens:
        for split in splits:
            csv_split_path = os.path.join(dirname, f'{name}_{split}{ext}')
            assert os.path.exists(csv_split_path)
            print(f"Generating file for seq_len: {seq_len}, split: {split}")
            cache_dataset(midi_root, csv_split_path, seq_len, split)


if __name__ == '__main__':
    main()
