# AI_final_project_team_63

# MIDI Genre & Emotion Classifier

This is a submodule of the Interactive Music Composition, providing automatic genre and emotion classification for MIDI music files.

## What is this?

The MIDI classifier takes a `.midi` file and predicts both its **genre** and **emotion** using a lightweight CNN-based deep learning model.

- **Input**: Single MIDI file (`.midi` or `.mid`)
- **Output**: Predicted genre and emotion, with probability scores for each class

## How to Train

### 1. Install dependencies

```bash
cd midi_classifier
pip install -r requirements.txt
```

### 2. Data Preprocessing
Before training or prediction, you must preprocess your MIDI data to extract features and generate numpy arrays.

```bash
python preprocess.py
```
- Place your raw MIDI files in the data/ folder (default).

- Preprocessed files will be saved in the preprocessed/ directory (including .npy features and a meta.csv metadata file).

#### Dataset Format & Label Extraction
The preprocessing script automatically extracts emotion and genre labels from each MIDI file’s name.
The expected filename format is:
```php-template
XMIDI_<Emotion>_<Genre>_<ID>.midi
```

### 3. (Optional) Train your own model

By default, you can use the provided pre-trained model.
To retrain:

```bash
python train.py
```

- Place your MIDI files in the data/ folder.
- Training progress and final model will be saved as midi_multitask_cnn.pt.

### 4. Predict Genre and Emotion

```bash
python predict.py
```
## Integration

Due to GitHub’s file size limit (100 MB), the pre-trained model weights (.pt file) are not included directly in this repository.
You can download the latest pre-trained checkpoint from Google Drive:
https://drive.google.com/file/d/1d2n1nMdNuplPc7bI5ftxdrVIXGM9JTzz/view?usp=sharing

After downloading, please place the file in your midi_classifier/ directory (next to train.py) and rename it as ''midi_multitask_cnn_final.pt''.

To call the classifier from another Python module:

```python
from midi_classifier.predict import predict_emotion_genre_prob

emotion, genre, emotion_probs, genre_probs = predict_emotion_genre_prob("path/to/file.midi")
```