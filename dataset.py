# dataset.py
import torch
from torch.utils.data import Dataset

class MusicDataset(Dataset):
    """
    用於條件式音樂生成的 PyTorch Dataset。
    每筆資料包含:
      - tokens: 音樂事件序列 (List[int])，自動裁剪或 padding 至 max_len
      - emotion: 情緒標籤 index (int)
      - genre: 種類標籤 index (int)
      - length: 原始 token 長度 (int)，可用於 mask
    """
    def __init__(self, data, emo2idx, gen2idx, max_len=512, pad_token=0):
        self.data = data
        self.emo2idx = emo2idx
        self.gen2idx = gen2idx
        self.max_len = max_len
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        # 裁剪或 padding
        tokens = tokens[:self.max_len]
        length = len(tokens)
        if length < self.max_len:
            tokens = tokens + [self.pad_token] * (self.max_len - length)
        emotion = self.emo2idx[item['emotion']]
        genre = self.gen2idx[item['genre']]
        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(emotion, dtype=torch.long),
            torch.tensor(genre, dtype=torch.long),
            length
        )
