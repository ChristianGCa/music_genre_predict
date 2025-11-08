import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pydub import AudioSegment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)
SR = 22050
MODEL_DIR = "./models"

class CNNModel(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(64*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def convert_to_wav(file_path):
    if file_path.lower().endswith(".mp3"):
        wav_path = file_path.rsplit(".",1)[0] + ".wav"
        audio = AudioSegment.from_mp3(file_path)
        audio.export(wav_path, format="wav")
        return wav_path
    return file_path

def extract_middle_segment(wav_path, duration=30, sr=SR):
    y, _ = librosa.load(wav_path, sr=sr)
    total_samples = len(y)
    seg_samples = duration * sr
    start = max(0, total_samples // 2 - seg_samples // 2)
    end = start + seg_samples
    return y[start:end]

def split_into_segments(y, segment_sec=3, sr=SR):
    seg_len = segment_sec * sr
    segments = []
    for start in range(0, len(y), seg_len):
        segment = y[start:start+seg_len]
        if len(segment) == seg_len:
            segments.append(segment)
    return segments

def spectrogram_from_audio(y, sr=SR):
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(2,2))
    librosa.display.specshow(S_db, sr=sr, cmap='gray_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("temp.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    img = Image.open("temp.png").convert('L')
    img = img.resize(IMG_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0)

def predict_genre(model_path, audio_segments):
    model = CNNModel(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    probs_list = []

    with torch.no_grad():
        for seg in audio_segments:
            probs = model(seg.to(DEVICE))
            probs = torch.softmax(probs, dim=1)
            probs_list.append(probs.cpu().numpy())

    probs_array = np.vstack(probs_list)
    mean_probs = probs_array.mean(axis=0)
    return mean_probs

genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

def main(audio_file):

    wav_file = convert_to_wav(audio_file)

    y_30s = extract_middle_segment(wav_file, duration=30)
    seg_30s_tensor = [spectrogram_from_audio(y_30s)]

    segments_3s = split_into_segments(y_30s, segment_sec=3)
    segments_3s_tensors = [spectrogram_from_audio(seg) for seg in segments_3s]

    probs_30s = predict_genre(os.path.join(MODEL_DIR, "cnn_30s.pth"), seg_30s_tensor)

    probs_3s = predict_genre(os.path.join(MODEL_DIR, "cnn_3s.pth"), segments_3s_tensors)


    print(f"\nPredição usando CNN 30s:")
    sorted_30s = sorted(zip(genres, probs_30s), key=lambda x: x[1], reverse=True)
    for g, p in sorted_30s:
        print(f"{g}: {p*100:.2f}%")

    print(f"\nPredição usando CNN 3s:")
    sorted_3s = sorted(zip(genres, probs_3s), key=lambda x: x[1], reverse=True)
    for g, p in sorted_3s:
        print(f"{g}: {p*100:.2f}%")


if __name__ == "__main__":

    audio_file_path = "/home/christian/Músicas/music2.mp3"
    #audio_file_path = "/home/christian/Músicas/music5.mp3"
    main(audio_file_path)
