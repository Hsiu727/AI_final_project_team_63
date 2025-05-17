import pandas as pd
import clip
import torch
import numpy as np
from tqdm import tqdm

class CLIPTextEncoder:
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode(self, texts, batch_size=32):
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i+batch_size]
                tokens = clip.tokenize(batch).to(self.device)
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_embeddings.append(features.cpu().numpy())
        return np.vstack(all_embeddings)

    def encode_and_save(self, csv_path, desc_column="description", out_path="clip_text_embeddings.npy"):
        df = pd.read_csv(csv_path)
        texts = df[desc_column].astype(str).tolist()
        embeddings = self.encode(texts)
        np.save(out_path, embeddings)
        print(f"儲存完畢：{out_path}，shape={embeddings.shape}")
        return embeddings
