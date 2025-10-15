import os

from data_pipeline import MIDIDataset


def cache_dataset(midi_root: os.PathLike, csv_path: os.PathLike, seq_len: int=128) -> None:
    MIDIDataset(
        midi_root=midi_root,
        seq_len=seq_len,
        fs=16,
        cache=True,
        label_mode="csv",
        csv_path=csv_path,
        csv_path_column="path",
        csv_label_column="genre_id",
    )


def main():
    midi_root = "dataset/data/Lakh_MIDI_Dataset_Clean"
    csv_path  = "dataset/data/lakh_clean_merged_homologado.csv"
    seq_lens = [32, 64, 128]
    for seq_len in seq_lens:
        cache_dataset(midi_root, csv_path, seq_len)


if __name__ == '__main__':
    main()
