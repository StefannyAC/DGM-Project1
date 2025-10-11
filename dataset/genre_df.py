
import os
import glob
from typing import List

import pandas as pd
from tqdm import tqdm

from msongsdb import hdf5_getters as hdf5


def find_all_metadata_files(dir_path: os.PathLike) -> List[os.PathLike]:
    all_metadata_files = glob.glob(f"{dir_path}/**/*.h5", recursive=True)
    return all_metadata_files
    all_metadata_files = [
        os.path.join(dir_path, path)
        for path in all_metadata_files
    ]
    return all_metadata_files


def read_metadata_file(fpath: os.PathLike) -> dict:
    h5 = hdf5.open_h5_file_read(fpath)
    return {
        'artist': hdf5.get_artist_name(h5).decode('utf-8'),
        'title': hdf5.get_title(h5).decode('utf-8'),
        'mbtags': [
            tag.decode('utf-8')
            for tag in hdf5.get_artist_mbtags(h5)
        ],
        'year': hdf5.get_year(h5)
    }
    h5.close()


def main():
    metadata_files = find_all_metadata_files("data/lmd_matched_h5/")
    metadata_per_file = []
    for mdfile in tqdm(metadata_files):
        metadata_per_file.append(read_metadata_file(mdfile))
    df = pd.DataFrame(metadata_per_file)
    df = df.sort_values(by=['artist', 'title'])
    df.to_csv("metadata.csv")


if __name__ == '__main__':
    main()
