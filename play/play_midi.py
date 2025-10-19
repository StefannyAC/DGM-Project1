# play_midi.py
# ============================================================
# Realizado por: Emmanuel Larralde y Stefanny Arboleda
# Proyecto # 1 - Modelos Generativos Profundos 
# Artículo base: "Design of an Improved Model for Music Sequence Generation Using Conditional Variational Autoencoder and Conditional GAN"
# ============================================================
# Este script sirve para audio WAV (o un objeto PrettyMIDI)
# a partir de un piano-roll binario (128 x T). También muestra un
# ejemplo mínimo de cómo tomar una muestra desde el MIDIDataset
# y sintetizarla como audio de ondas seno. En general no se usa, pero se deja por si se quisiera probar.
# ============================================================
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
    Función para convertir un piano-roll binario (128 x T) a un objeto PrettyMIDI.

    Args:
        piano_roll: Matriz binaria (128, T) donde cada fila es un número de nota MIDI y
            cada columna un frame temporal; valores 0/1 indican nota apagada/encendida.
        fs: Frecuencia de muestreo del piano-roll en frames por segundo (no Hz de audio).
        program: Número de instrumento MIDI (0 = Acoustic Grand Piano). Se usa un solo instrumento.

    Returns:
        pretty_midi.PrettyMIDI: Objeto PrettyMIDI con una sola pista (instrumento) y notas construidas
        detectando regiones contiguas de 1s por cada nota.
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    
    # Recorre las 128 notas MIDI (0..127).
    for note_number in range(128):
        # Extrae la fila correspondiente a esa nota (vector de longitud T con 0/1).
        piano_roll_note = piano_roll[note_number]
        # Padded con ceros en ambos extremos para detectar transiciones con diff.
        padded = np.pad(piano_roll_note, (1, 1), mode='constant')
        # Derivada discreta: +1 marca inicio, -1 marca fin de regiones en 1.
        diff = np.diff(padded)
        # Índices de inicio/fin (en frames) -> conviértelos a segundos dividiendo por fs.
        start_indices = np.where(diff == 1)[0] / fs 
        end_indices = np.where(diff == -1)[0] / fs
        
        # Construye notas (pitch fijo = note_number; velocidad fija = 100).
        for start, end in zip(start_indices, end_indices):
            note = pretty_midi.Note(
                velocity=100,  # can tweak
                pitch=note_number,
                start=start,
                end=end
            )
            instrument.notes.append(note)
    
    midi.instruments.append(instrument) # Inserta el instrumento con todas sus notas.
    return midi

def piano_roll_to_audio(piano_roll, fs_roll=16, fs_audio=44100, duration=None):
    """
    Función para sintetizar **audio** a partir de un piano-roll binario usando **ondas seno** simples.

    Args:
        piano_roll (np.ndarray): Matriz binaria (128, T) donde 1 indica que la nota (fila) está activa.
        fs_roll (int): Frames por segundo del piano-roll (p. ej., 16 fps).
        fs_audio (int): Frecuencia de muestreo objetivo del audio (p. ej., 44100 Hz).
        duration (float | None): Duración total en segundos; si None, se infiere como T / fs_roll.

    Returns:
        np.ndarray: Señal de audio mono en `float64` dentro de [-1, 1] (normalizada).
    """
    n_notes, T = piano_roll.shape
    if duration is None:
        duration = T / fs_roll # Duración total en segundos
    
    # Buffer de audio mono inicializado en cero (float64 para sumar con menos error acumulado)
    audio = np.zeros(int(duration * fs_audio))
    
    # Conversión de nota MIDI -> frecuencia en Hz (temple igual)
    def midi_to_freq(midi_note):
        return 440.0 * 2**((midi_note - 69) / 12)
    
    # Recorre todas las notas MIDI
    for note_number in range(n_notes):
        # Indices (en frames) donde la nota está activa (1)
        indices = np.where(piano_roll[note_number] > 0)[0]
        if len(indices) == 0:
            continue # Nada que sintetizar para esta nota
        # Detecta regiones contiguas de 1s
        padded = np.pad(piano_roll[note_number], (1,1))
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0] # inicios (en frames)
        ends = np.where(diff == -1)[0] # finales (en frames)
        
        # Para cada región activa, suma una senoide a la frecuencia de la nota
        for start, end in zip(starts, ends):
            start_sample = int(start / fs_roll * fs_audio) # frame->muestra
            end_sample = int(end / fs_roll * fs_audio)
            t = np.arange(end_sample - start_sample) / fs_audio
            freq = midi_to_freq(note_number)
            # Senoide simple (amplitud 0.2 para evitar clipping en combinación)
            audio[start_sample:end_sample] += 0.2 * np.sin(2 * np.pi * freq * t)
    
    audio /= np.max(np.abs(audio))
    return audio

def get_dataset(seq_len = 128):
    return MIDIDataset(MIDI_ROOT, seq_len, csv_path=CSV_PATH)

def main():
    # Se deja ejemplo puntual por si se quiere ejecutar
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