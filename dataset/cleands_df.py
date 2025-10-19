# prepare_midi_dataset.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Extrae artista título y path del archivo midi. No usar salvo se necesite reconstruir el dataset.
# ============================================================
import os
import glob

import pandas as pd
from tqdm import tqdm


def find_all_midi_files(dir_path: os.PathLike):
    all_midi_files = glob.glob(f'{dir_path}/**/*.mid', recursive=True)

    return all_midi_files


def main():
    dirpath = 'data/lakh-midi-clean/'
    dirpath = os.path.abspath(dirpath)
    midi_files = find_all_midi_files(dirpath)

    fields = []
    for midi_f in tqdm(midi_files):
        dirname = os.path.basename(os.path.dirname(midi_f))
        artist = " ".join(dirname.split('_'))

        basename = os.path.basename(midi_f)
        fnmae_no_ext = basename.split('.')[0]
        song = " ".join(fnmae_no_ext.split('_'))

        path = midi_f.removeprefix(dirpath)
        try:
            path = path.removeprefix('/')
        except:
            pass

        info = {
            'artist': artist,
            'title': song,
            'path': path   
        }

        fields.append(info)
    
    df = pd.DataFrame(fields)
    df = df.sort_values(by=['artist', 'title'])

    df.to_csv("lakh_clean.csv")


if __name__ == '__main__':
    main()
        