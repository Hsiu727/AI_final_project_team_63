import torch
import torch.nn as nn
import timm
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiTaskMIDIMHAttention(nn.Module):
    def __init__(self, num_emotions, num_genres, max_length=500, 
                 in_dim=128, d_model=512, nhead=8, num_layers=8, dropout=0.2):
        super().__init__()
        self.embedding = nn.Linear(in_dim, d_model)
        self.posenc = PositionalEncoding(d_model, max_length+1) # +1 for [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,   # feedforward 更大
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)    # 加 LayerNorm 穩定
        self.head_emotion = nn.Linear(d_model, num_emotions)
        self.head_genre = nn.Linear(d_model, num_genres)

    def forward(self, x):
        B = x.size(0)
        x = x.permute(0, 2, 1)                  # [B, T, in_dim]
        x = self.embedding(x)                    # [B, T, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)         # [B, T+1, d_model]
        x = self.posenc(x)
        x = self.transformer(x)                  # [B, T+1, d_model]
        cls_out = self.norm(x[:, 0, :])          # [B, d_model]
        cls_out = self.dropout(cls_out)
        out_emotion = self.head_emotion(cls_out)
        out_genre = self.head_genre(cls_out)
        return out_emotion, out_genre

class MultiTaskMIDIConvNeXt(nn.Module):
    def __init__(self, num_emotions, num_genres, max_length=500):
        super().__init__()
        # ConvNeXt-tiny，支援輸入通道1
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=False,
            in_chans=1,
            num_classes=0  # 不用 backbone 的 head
        )
        # ConvNeXt 對應最後展開特徵維度
        self.feat_dim = self.backbone.num_features
        # 兩個 head
        self.head_emotion = nn.Linear(self.feat_dim, num_emotions)
        self.head_genre = nn.Linear(self.feat_dim, num_genres)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, 128, max_length]
        # print("Input x:", x.shape)
        f = self.backbone.forward_features(x)
        # print("Backbone output f:", f.shape)
        if f.ndim == 4:
            f = f.mean([-2, -1])  # [B, C]
        out_emotion = self.head_emotion(f)
        out_genre = self.head_genre(f)
        return out_emotion, out_genre

class MultiTaskMIDICNN(nn.Module):
    def __init__(self, num_emotions, num_genres, max_length=500):
        super().__init__()
        # 強化 CNN 結構: 更深更寬
        self.conv1 = nn.Conv2d(1, 32, (8, 5), padding=(3,2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (6, 3), padding=(2,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, (4, 3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d((2,2))
        self.dropout = nn.Dropout(0.3)
        # 動態計算展開大小
        test = torch.zeros((1, 1, 128, max_length))
        with torch.no_grad():
            x = self.pool(self.bn1(self.conv1(test)))
            x = self.pool(self.bn2(self.conv2(x)))
            x = self.pool(self.bn3(self.conv3(x)))
            flat_size = x.view(1, -1).shape[1]
        self.fc = nn.Linear(flat_size, 256)
        self.head_emotion = nn.Linear(256, num_emotions)
        self.head_genre = nn.Linear(256, num_genres)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        # 【重點：這裡不要 relu】
        x = self.dropout(self.fc(x))
        out_emotion = self.head_emotion(x)
        out_genre = self.head_genre(x)
        return out_emotion, out_genre
