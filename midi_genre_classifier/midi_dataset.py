import os
from torch.utils.data import Dataset
from utils import midi_to_pianoroll

def extract_emotion_genre_from_filename(filename):
    # Ex: XMIDI_Happy_Jazz_00000001.midi
    parts = filename.split('_')
    emotion = parts[1]
    genre = parts[2]
    return emotion, genre

class MIDIMultiLabelDataset(Dataset):
    def __init__(self, root_dir, max_length=500):
        self.samples = []
        self.emotions = set()
        self.genres = set()
        for f in os.listdir(root_dir):
            if f.lower().endswith('.midi'):
                emotion, genre = extract_emotion_genre_from_filename(f)
                self.emotions.add(emotion)
                self.genres.add(genre)
        self.emotion2idx = {e: i for i, e in enumerate(sorted(self.emotions))}
        self.genre2idx = {g: i for i, g in enumerate(sorted(self.genres))}
        # Second pass to build samples
        self.labels = []
        for f in os.listdir(root_dir):
            if f.lower().endswith('.midi'):
                emotion, genre = extract_emotion_genre_from_filename(f)
                self.samples.append(os.path.join(root_dir, f))
                self.labels.append((
                    self.emotion2idx[emotion],
                    self.genre2idx[genre]
                ))
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = midi_to_pianoroll(self.samples[idx], max_length=self.max_length)
        emotion_label, genre_label = self.labels[idx]
        return x, emotion_label, genre_label
