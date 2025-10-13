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