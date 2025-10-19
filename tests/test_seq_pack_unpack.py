# test_seq_pack_unpack.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este script prueba las funciones de empaquetado y desempaquetado
# de matrices tipo piano-roll para asegurar que son inversas una de la otra. 
# Es un script simple que genera matrices aleatorias, las empaqueta y luego las desempaqueta, no usarlo pues no es parte del flujo principal del proyecto.
# ============================================================
import numpy as np
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline import pack_matrix, unpack_matrix


def main():
    for _ in tqdm(range(1000)):
        mat = np.random.choice(a=[0, 1], size=(128, 128), p=[0.1, 0.9])
        mat = mat.astype(np.uint8)
        mat_2 = unpack_matrix(pack_matrix(mat))
        assert (mat == mat_2).all(), "Errro"


if __name__ == '__main__':
    main()