import torch
from torch.utils.data import Dataset
import pickle

class FullBandMusicDataset(Dataset):
    def __init__(self, data_path, emo2idx_path, gen2idx_path, max_len=512, pad_token=0):
        # 載入前處理好的資料
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(emo2idx_path, "rb") as f:
            self.emo2idx = pickle.load(f)
        with open(gen2idx_path, "rb") as f:
            self.gen2idx = pickle.load(f)
        self.max_len = max_len
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        # truncate
        tokens = tokens[:self.max_len]
        length = len(tokens)
        # pad if needed
        if length < self.max_len:
            tokens += [self.pad_token] * (self.max_len - length)
        emotion = self.emo2idx[item['emotion']]
        genre = self.gen2idx[item['genre']]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(emotion, dtype=torch.long),
            torch.tensor(genre, dtype=torch.long),
            length
        )

# 如果你要測試 DataLoader：
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = FullBandMusicDataset(
        "dataset_fullband.pkl", "emo2idx.pkl", "gen2idx.pkl",
        max_len=512, pad_token=0
    )
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    for batch in dl:
        tokens, emos, gens, lengths = batch
        print("tokens:", tokens.shape)
        print("emos:", emos)
        print("gens:", gens)
        print("lengths:", lengths)
        break
