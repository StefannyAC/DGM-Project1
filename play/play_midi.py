import os, sys

import pretty_midi
import numpy as np
import pandas as pd
from scipy.io.wavfile import write

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_pipeline import MIDI_ROOT, MIDIDataset, CSV_PATH, midi_to_piano_roll


FLUID_GM_PATH = os.path.join(os.path.dirname(__file__), "FluidR3_GM", "FluidR3_GM.sf2")


def piano_roll_to_pretty_midi(piano_roll, fs=16, program=0):
    """
    Convert a piano roll (binary) to a PrettyMIDI object.
    
    piano_roll: (128, T) binary numpy array
    fs: sampling frequency used in piano_roll (frames per second)
    program: MIDI instrument number (0=piano)
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    # iterate over all notes
    for note_number in range(128):
        # find contiguous regions where note is active
        piano_roll_note = piano_roll[note_number]
        # pad with zeros to detect edges
        padded = np.pad(piano_roll_note, (1, 1), mode='constant')
        diff = np.diff(padded)
        start_indices = np.where(diff == 1)[0] / fs
        end_indices = np.where(diff == -1)[0] / fs
        
        for start, end in zip(start_indices, end_indices):
            note = pretty_midi.Note(
                velocity=100,  # can tweak
                pitch=note_number,
                start=start,
                end=end
            )
            instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    return midi

def piano_roll_to_audio(piano_roll, fs_roll=16, fs_audio=44100, duration=None):
    """
    Convert a binary piano roll to raw audio (sine waves).
    
    piano_roll: (128, T) binary array
    fs_roll: frame rate of piano roll
    fs_audio: desired audio sample rate
    duration: optional, total duration in seconds
    """
    n_notes, T = piano_roll.shape
    if duration is None:
        duration = T / fs_roll
    
    audio = np.zeros(int(duration * fs_audio))
    
    # MIDI note to frequency
    def midi_to_freq(midi_note):
        return 440.0 * 2**((midi_note - 69) / 12)
    
    # iterate over notes
    for note_number in range(n_notes):
        indices = np.where(piano_roll[note_number] > 0)[0]
        if len(indices) == 0:
            continue
        # find contiguous regions of ones
        padded = np.pad(piano_roll[note_number], (1,1))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            start_sample = int(start / fs_roll * fs_audio)
            end_sample = int(end / fs_roll * fs_audio)
            t = np.arange(end_sample - start_sample) / fs_audio
            freq = midi_to_freq(note_number)
            # simple sine wave
            audio[start_sample:end_sample] += 0.2 * np.sin(2 * np.pi * freq * t)
    
    # normalize to prevent clipping
    audio /= np.max(np.abs(audio))
    return audio


def get_dataset():
    return MIDIDataset(MIDI_ROOT, 128, csv_path=CSV_PATH)


def main():
    #df = pd.read_csv(CSV_PATH)
    #daft_punk = df[df['artist'] == 'Daft Punk']
    #midi_path = os.path.join(MIDI_ROOT, daft_punk['path'].iloc[3])
    #binary_roll = midi_to_piano_roll(midi_path).astype(float)

    dataset = get_dataset()
    binary_roll = dataset[0]['piano_roll'].numpy().astype(float)
    
    audio = piano_roll_to_audio(binary_roll, fs_roll=16)
    write('output.wav', 44100, (audio * 32767).astype(np.int16))



if __name__ == '__main__':
    main()
