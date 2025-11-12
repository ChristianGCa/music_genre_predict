import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

# Permite importar o audio_utils.py
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from audio_utils import convert_to_wav, extract_middle_segment, save_mel_spectrogram, load_spectrogram_as_tensor


# ---------------------------
# CONFIGURAÇÕES
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminho do áudio a ser analisado
AUDIO_PATH = "/home/christian/Músicas/music1.mp3"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

MODEL_PATH = "./models/spectrogram_model.pth"
TEMP_SPEC_PATH = "./temp_spec.png"


# ---------------------------
# DEFINIÇÃO DO MODELO CNN
# (deve ser igual ao usado no treino)
# ---------------------------
class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpectrogramCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# FUNÇÕES DE UTILIDADE
# ---------------------------
def load_model(model_path, num_classes):
    model = SpectrogramCNN(num_classes=num_classes).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_genre(model, tensor):
    with torch.no_grad():
        outputs = model(tensor.to(DEVICE))
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs


# ---------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------
def main(audio_path):
    print(f"🎵 Processando: {audio_path}")
    wav_path = convert_to_wav(audio_path)
    y = extract_middle_segment(wav_path, duration=3.0)
    save_mel_spectrogram(y, out_path=TEMP_SPEC_PATH)

    tensor = load_spectrogram_as_tensor(TEMP_SPEC_PATH)

    print("🔄 Carregando modelo...")
    model = load_model(MODEL_PATH, num_classes=len(GENRES))

    print("🎯 Realizando predição...")
    probs = predict_genre(model, tensor)

    print("\n🎶 Probabilidades por gênero:")
    for genre, p in zip(GENRES, probs):
        print(f" - {genre:10s}: {p*100:6.2f}%")

    pred_idx = np.argmax(probs)
    print(f"\n✅ Gênero mais provável: **{GENRES[pred_idx].upper()}** ({probs[pred_idx]*100:.2f}%)")


# ---------------------------
# RODAR DIRETAMENTE
# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(AUDIO_PATH):
        print(f"❌ Erro: arquivo não encontrado -> {AUDIO_PATH}")
    else:
        main(AUDIO_PATH)
