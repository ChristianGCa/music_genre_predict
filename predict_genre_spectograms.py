import os
import torch
import torch.nn as nn
import numpy as np
from scripts.audio_utils import (
    convert_to_wav,
    extract_middle_segment,
    split_segments,
    save_mel_spectrogram,
    load_spectrogram_as_tensor,
    SR,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)
MODEL_DIR = "./models"
TEMP_IMG = "temp_spectrogram.png"

GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]


class CNNModel(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def create_spectrogram_tensor(y, sr=SR):
    """Cria e carrega tensor de espectrograma usando funções do audio_utils."""
    save_mel_spectrogram(y, sr=sr, out_path=TEMP_IMG)
    return load_spectrogram_as_tensor(TEMP_IMG, IMG_SIZE)


def predict_genre(model_path, spectrogram_tensors):
    """Prediz o gênero médio a partir de múltiplos espectrogramas."""
    model = CNNModel(num_classes=len(GENRES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    probs_list = []
    with torch.no_grad():
        for tensor in spectrogram_tensors:
            tensor = tensor.to(DEVICE)
            probs = torch.softmax(model(tensor), dim=1)
            probs_list.append(probs.cpu().numpy())

    mean_probs = np.vstack(probs_list).mean(axis=0)
    return mean_probs

def main(audio_file):
    wav_file = convert_to_wav(audio_file)

    y_30s = extract_middle_segment(wav_file, duration=30)
    seg_30s_tensor = [create_spectrogram_tensor(y_30s)]

    segments_3s = split_segments(y_30s, sr=SR, segment_sec=3)
    seg_3s_tensors = [create_spectrogram_tensor(seg) for seg in segments_3s]

    probs_30s = predict_genre(os.path.join(MODEL_DIR, "cnn_30s.pth"), seg_30s_tensor)
    probs_3s = predict_genre(os.path.join(MODEL_DIR, "cnn_3s.pth"), seg_3s_tensors)

    print("\nModelo cnn_30s:")
    for g, p in sorted(zip(GENRES, probs_30s), key=lambda x: x[1], reverse=True):
        print(f"{g:10s}: {p*100:.2f}%")

    print("\nModelo cnn_3s:")
    for g, p in sorted(zip(GENRES, probs_3s), key=lambda x: x[1], reverse=True):
        print(f"{g:10s}: {p*100:.2f}%")


if __name__ == "__main__":
    audio_file_path = "/home/christian/Músicas/music1.mp3"
    main(audio_file_path)
