# split_dataset_csv.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este script crea splits de entrenamiento, validacion y prueba y guardar los csv con los datos para reproducibilidad. No es necesario volverlo a ejecutar.
# ============================================================
import os

import pandas as  pd


csv_path  = "dataset/data/lakh_clean_merged_homologado.csv"
train_r = 0.8
val_r = 0.1
test_r = 1.0 - train_r - val_r


def main():
    
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    genres = df['genre_id'].unique()
    sub_dfs = [
        df[df['genre_id'] == id]
        for id in genres
    ]

    dirname, basename = os.path.split(csv_path)
    name, ext = os.path.splitext(basename)

    train_dfs = []
    val_dfs = []
    test_dfs = []
    for sub_df in sub_dfs:
        n = len(sub_df)
        train_n = int(n * train_r)
        val_n = int(n * val_r)

        train_dfs.append(sub_df.iloc[:train_n])
        val_dfs.append(sub_df.iloc[train_n: train_n + val_n])
        test_dfs.append(sub_df.iloc[train_n + val_n:])

    train_df = pd.concat(train_dfs).sort_values(by=['artist', 'title'])
    val_df = pd.concat(val_dfs).sort_values(by=['artist', 'title'])
    test_df = pd.concat(test_dfs).sort_values(by=['artist', 'title'])

    train_df.to_csv(os.path.join(dirname, f'{name}_train{ext}'))
    val_df.to_csv(os.path.join(dirname, f'{name}_val{ext}'))
    test_df.to_csv(os.path.join(dirname, f'{name}_test{ext}'))


        


if __name__ == '__main__':
    main()
