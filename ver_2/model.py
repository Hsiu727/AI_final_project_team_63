import torch
import torch.nn as nn

class CondTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        emo_num,
        gen_num,
        event_type_num,
        d_model=256,
        nhead=8,
        nlayers=4,
        max_seq_len=512,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emo_emb   = nn.Embedding(emo_num, d_model)
        self.gen_emb   = nn.Embedding(gen_num, d_model)
        self.event_type_emb = nn.Embedding(event_type_num, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, event_types, emotion, genre):
        """
        tokens:      [B, L]
        event_types: [B, L]
        emotion, genre: [B]
        """
        B, L = tokens.size()
        device = tokens.device

        tok = self.token_emb(tokens)                               # [B, L, d_model]
        pos = self.pos_emb(torch.arange(L, device=device)).unsqueeze(0).expand(B, L, -1)
        type_emb = self.event_type_emb(event_types)                # [B, L, d_model]
        emo = self.emo_emb(emotion).unsqueeze(1).expand(B, L, -1)
        gen = self.gen_emb(genre).unsqueeze(1).expand(B, L, -1)
        x = tok + pos + type_emb + emo + gen                      # [B, L, d_model]
        x = x.transpose(0, 1)                                     # [L, B, d_model]
        mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        out = self.transformer(x, mask)
        out = self.fc(out).transpose(0, 1)                        # [B, L, vocab_size]
        return out
