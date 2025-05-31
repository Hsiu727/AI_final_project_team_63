import torch
import torch.nn as nn

class CondTransformer(nn.Module):
    """
    條件式自回歸 Transformer 模型

    Inputs:
      - tokens: LongTensor of shape [B, L] (token ID sequences)
      - emotion: LongTensor of shape [B] (emotion label indices)
      - genre:   LongTensor of shape [B] (genre label indices)

    Output:
      - logits: FloatTensor of shape [B, L, vocab_size]
    """
    def __init__(
        self,
        vocab_size: int,
        emo_num: int,
        gen_num: int,
        d_model: int = 256,
        nhead: int = 8,
        nlayers: int = 4,
        max_seq_len: int = 512,
        attn_dropout: float = 0.0,
        use_layernorm: bool = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emo_emb   = nn.Embedding(emo_num, d_model)
        self.gen_emb   = nn.Embedding(gen_num, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=attn_dropout,
            norm_first=use_layernorm # only for PyTorch >=1.11
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.fc = nn.Linear(d_model, vocab_size)
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.LongTensor, emotion: torch.LongTensor, genre: torch.LongTensor):
        B, L = tokens.size()
        device = tokens.device
        tok = self.token_emb(tokens)
        pos = self.pos_emb(torch.arange(L, device=device)).unsqueeze(0).expand(B, -1, -1)
        emo = self.emo_emb(emotion).unsqueeze(1).expand(B, L, -1)
        gen = self.gen_emb(genre).unsqueeze(1).expand(B, L, -1)
        x = tok + pos + emo + gen
        if self.use_layernorm:
            x = self.ln(x)
        x = x.transpose(0, 1)
        mask = torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)
        out = self.transformer(x, mask)
        out = self.fc(out)
        out = out.transpose(0, 1)
        return out

# Example usage:
# model = CondTransformer(vocab_size=500, emo_num=5, gen_num=8)
# logits = model(tokens_batch, emotion_batch, genre_batch)  # [B, L, vocab_size]
