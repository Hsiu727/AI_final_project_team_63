import torch
from model import MultiTaskMIDICNN
from utils import midi_to_pianoroll

def predict_emotion_genre_prob(filepath, model_path="midi_multitask_cnn.pt", max_length=500):
    ckpt = torch.load(model_path, map_location='cpu')
    emotion2idx = ckpt['emotion2idx']
    genre2idx = ckpt['genre2idx']
    idx2emotion = {v: k for k, v in emotion2idx.items()}
    idx2genre = {v: k for k, v in genre2idx.items()}
    num_emotions = len(emotion2idx)
    num_genres = len(genre2idx)
    model = MultiTaskMIDICNN(num_emotions, num_genres, max_length)
    model.load_state_dict(ckpt['model'])
    model.eval()

    x = midi_to_pianoroll(filepath, max_length)
    x = torch.tensor(x).unsqueeze(0)
    with torch.no_grad():
        out_emo, out_gen = model(x)
        prob_emo = torch.softmax(out_emo, dim=1).squeeze().cpu().numpy()
        prob_gen = torch.softmax(out_gen, dim=1).squeeze().cpu().numpy()
        top_emo = prob_emo.argmax()
        top_gen = prob_gen.argmax()

        # 印出所有 emotion 百分比
        print("=== Emotion Probabilities ===")
        for idx, prob in enumerate(prob_emo):
            print(f"{idx2emotion[idx]}: {prob*100:.2f}%")
        print(f"Predicted Emotion: {idx2emotion[top_emo]}\n")

        # 印出所有 genre 百分比
        print("=== Genre Probabilities ===")
        for idx, prob in enumerate(prob_gen):
            print(f"{idx2genre[idx]}: {prob*100:.2f}%")
        print(f"Predicted Genre: {idx2genre[top_gen]}")

        return idx2emotion[top_emo], idx2genre[top_gen], prob_emo, prob_gen

if __name__ == '__main__':
    path = "data/XMIDI_quiet_classical_45LARDVD.midi"
    predict_emotion_genre_prob(path)
