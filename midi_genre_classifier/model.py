import torch
import torch.nn as nn

class MultiTaskMIDICNN(nn.Module):
    def __init__(self, num_emotions, num_genres, max_length=500):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (12, 5), stride=1)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, (6, 3), stride=1)
        self.pool2 = nn.MaxPool2d((2, 2))
        # 動態計算展開大小
        test = torch.zeros((1, 1, 128, max_length))
        with torch.no_grad():
            x = self.pool1(self.conv1(test))
            x = self.pool2(self.conv2(x))
            flat_size = x.view(1, -1).shape[1]
        self.fc = nn.Linear(flat_size, 128)
        self.head_emotion = nn.Linear(128, num_emotions)
        self.head_genre = nn.Linear(128, num_genres)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        out_emotion = self.head_emotion(x)
        out_genre = self.head_genre(x)
        return out_emotion, out_genre
