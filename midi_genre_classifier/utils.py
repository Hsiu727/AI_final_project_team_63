import pretty_midi
import numpy as np

def midi_to_pianoroll(filepath, max_length=500):
    midi = pretty_midi.PrettyMIDI(filepath)
    piano_roll = midi.get_piano_roll(fs=4)  # 4 samples/sec
    # 固定長度
    if piano_roll.shape[1] > max_length:
        piano_roll = piano_roll[:, :max_length]
    else:
        pad_width = max_length - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad_width)))
    piano_roll = (piano_roll > 0).astype(np.float32)
    return piano_roll  # shape (128, max_length)
