# collate_fn.py
import torch
from torch.nn.utils.rnn import pad_sequence

def music_collate_fn(batch):
    """
    batch: list of tuples (tokens, emotion, genre, length)
      - tokens: 1D LongTensor of shape [seq_len]
      - emotion: LongTensor scalar
      - genre: LongTensor scalar
      - length: int
    回傳:
      - tokens_padded: LongTensor [B, L_max]
      - attention_mask: BoolTensor [B, L_max] (True = valid token)
      - emotions: LongTensor [B]
      - genres: LongTensor [B]
      - lengths: LongTensor [B]
    """
    # 拆開 batch
    tokens_list, emotions, genres, lengths = zip(*batch)

    # 動態 padding：將所有 tokens pad 到同一最大長度 L_max
    tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=0)  # [B, L_max]

    # attention mask (pad 位置為 False)
    attention_mask = tokens_padded.ne(0)  # [B, L_max]

    # stack 其它欄位
    emotions = torch.stack(emotions)      # [B]
    genres   = torch.stack(genres)        # [B]
    lengths  = torch.tensor(lengths)      # [B]

    return tokens_padded, attention_mask, emotions, genres, lengths
