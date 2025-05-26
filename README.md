# AI_final_project_team_63

# MIDI Genre & Emotion Classifier

This is a submodule of the Interactive Music Composition, providing automatic genre and emotion classification for MIDI music files.

## What is this?

The MIDI classifier takes a `.midi` file and predicts both its **genre** and **emotion** using a lightweight CNN-based deep learning model.

- **Input**: Single MIDI file (`.midi` or `.mid`)
- **Output**: Predicted genre and emotion, with probability scores for each class

## How to Use

### 1. Install dependencies

```bash
cd midi_classifier
pip install -r requirements.txt
```

### 2. (Optional) Train your own model

By default, you can use the provided pre-trained model.
To retrain:

```bash
python train.py
```

- Place your MIDI files in the data/ folder.
- Training progress and final model will be saved as midi_multitask_cnn.pt.

### 3. Predict Genre and Emotion

```bash
python predict.py
```
## Integration
To call the classifier from another Python module:

```python
from midi_classifier.predict import predict_emotion_genre_prob

emotion, genre, emotion_probs, genre_probs = predict_emotion_genre_prob("path/to/file.midi")
```