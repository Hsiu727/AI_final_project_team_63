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
        max_seq_len: int = 512
    ):
        super().__init__()
        # Embedding layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.emo_emb   = nn.Embedding(emo_num, d_model)
        self.gen_emb   = nn.Embedding(gen_num, d_model)

        # Transformer encoder-as-decoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Output projection
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.LongTensor, emotion: torch.LongTensor, genre: torch.LongTensor):
        """
        tokens: [B, L]
        emotion, genre: [B]
        Returns logits: [B, L, vocab_size]
        """
        B, L = tokens.size()
        device = tokens.device

        # Token + position embeddings
        tok = self.token_emb(tokens)                           # [B, L, d_model]
        pos = self.pos_emb(torch.arange(L, device=device))     # [L, d_model]
        pos = pos.unsqueeze(0).expand(B, -1, -1)               # [B, L, d_model]

        # Conditional embeddings
        emo = self.emo_emb(emotion).unsqueeze(1).expand(B, L, -1)  # [B, L, d_model]
        gen = self.gen_emb(genre).unsqueeze(1).expand(B, L, -1)   # [B, L, d_model]

        # Combine embeddings
        x = tok + pos + emo + gen            # [B, L, d_model]
        x = x.transpose(0, 1)                # [L, B, d_model]

        # Causal mask: prevent attention to future positions
        mask = torch.triu(
            torch.full((L, L), float('-inf'), device=device), diagonal=1
        )  # [L, L]

        # Transformer forward
        out = self.transformer(x, mask)      # [L, B, d_model]

        # Project to vocabulary
        out = self.fc(out)                   # [L, B, vocab_size]
        out = out.transpose(0, 1)            # [B, L, vocab_size]
        return out

# Example usage:
# model = CondTransformer(vocab_size=500, emo_num=5, gen_num=8)
# logits = model(tokens_batch, emotion_batch, genre_batch)  # [B, L, vocab_size]
