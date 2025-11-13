import os
import sys
import torch
import torch.nn as nn # Adicionado
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image # Adicionado para garantir a consistência com o treino

# Adiciona o diretório pai ao PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Importações do arquivo utilitário
from audio_utils import (
    convert_to_wav,
    load_audio,
    extract_middle_segment,
    split_segments,
    save_mel_spectrogram,
    load_spectrogram_as_tensor,
    SR,
)

# ----------------------------
# DEFINIÇÃO DA ARQUITETURA DO MODELO (TRANSFERIDA DO SCRIPT DE TREINO)
# É CRUCIAL QUE ISTO SEJA IDÊNTICO AO SCRIPT DE TREINO!
# ----------------------------
class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes):
        super(SpectrogramCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 512 -> 256
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 256 -> 128
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # ESTE TAMANHO É CRUCIAL (256 * 32 * 32)
            nn.Linear(256 * 32 * 32, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# ----------------------------


# ----------------------------
# CONFIGURAÇÕES
# ----------------------------
MUSIC_PATH = "/home/christian/Músicas/music1.mp3"
# Caminhos corrigidos para refletir o seu script de treino
MODEL_30S_PATH = "./models/cnn_30s.pth" 
MODEL_3S_PATH = "./models/cnn_3s.pth"
TMP_SPECTROGRAM = "./temp/temp_spectrogram.png"
IMAGE_SIZE = 512 # Mantém a consistência com o treino

os.makedirs("./temp", exist_ok=True)

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# FUNÇÃO: carregar modelo (CORRIGIDA)
# ----------------------------
def load_model(model_path):
    # 1. Instancia o modelo. O número de gêneros é len(GENRES).
    model = SpectrogramCNN(num_classes=len(GENRES)).to(DEVICE)
    
    # 2. Carrega o state_dict (o dicionário com os pesos)
    # Aqui é onde antes estava dando o erro de AttributeError
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # 3. Aplica os pesos no modelo instanciado
    model.load_state_dict(state_dict)
    
    # 4. Coloca o modelo em modo de avaliação (apenas para inferência)
    model.eval()
    
    return model

# ----------------------------
# FUNÇÃO: prever gênero (CORRIGIDA)
# ----------------------------
def predict_from_image(model, image_path):
    # 1. Define a transformação que imita a parte de teste do script de treino
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Redimensiona para 512x512
        transforms.ToTensor(),                       # Converte para Tensor (C x H x W)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normaliza
    ])

    # 2. Carrega e aplica a transformação
    # Nota: Abrimos a imagem como 'RGB' (3 canais) para corresponder ao modelo
    image = Image.open(image_path).convert('RGB') 
    tensor = test_transform(image).unsqueeze(0).to(DEVICE) # Adiciona a dimensão do batch [1, C, H, W]
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy().flatten()
    return probs

# ----------------------------
# FUNÇÃO PRINCIPAL
# ----------------------------
def main():
    # 1. Converter e carregar o áudio
    print(f"🎵 Carregando música: {MUSIC_PATH}")
    wav_path = convert_to_wav(MUSIC_PATH)
    y = load_audio(wav_path, sr=SR)

    # ----------------------------
    # Previsão com o modelo de 30s
    # ----------------------------
    print("\n🎧 Gerando espectrograma de 30s...")
    y_30s = extract_middle_segment(y, duration=30, sr=SR) # Usando 'y' (dados de áudio)
    save_mel_spectrogram(y_30s, sr=SR, out_path=TMP_SPECTROGRAM)

    try:
        model_30s = load_model(MODEL_30S_PATH)
    except Exception as e:
        print(f"\n❌ ERRO ao carregar o modelo 30s: {e}")
        # Aumentando a clareza sobre o erro de dimensionamento
        print("Verifique se a arquitetura SpectrogramCNN está correta ou se o arquivo do modelo existe.")
        return 

    probs_30s = predict_from_image(model_30s, TMP_SPECTROGRAM)

    print("\n🎶 Previsão (modelo 30s):")
    for genre, p in sorted(zip(GENRES, probs_30s), key=lambda x: -x[1]):
        print(f"{genre:10s}: {p*100:5.2f}%")

    # ----------------------------
    # Previsão com o modelo de 3s (média dos segmentos)
    # ----------------------------
    print("\n🎧 Gerando espectrogramas de 3s...")
    segments = split_segments(y, sr=SR, segment_sec=3)
    
    try:
        model_3s = load_model(MODEL_3S_PATH)
    except Exception as e:
        print(f"\n❌ ERRO ao carregar o modelo 3s: {e}")
        print("Verifique se a arquitetura SpectrogramCNN está correta ou se o arquivo do modelo existe.")
        return

    all_probs = []
    if segments:
        for i, chunk in enumerate(segments):
            tmp_path = f"./temp/temp_chunk_{i}.png" # Corrigido para usar a pasta 'temp'
            save_mel_spectrogram(chunk, sr=SR, out_path=tmp_path)
            probs = predict_from_image(model_3s, tmp_path)
            all_probs.append(probs)
            os.remove(tmp_path)
        
        mean_probs_3s = sum(all_probs) / len(all_probs)

        print("\n🎵 Previsão média (modelo 3s):")
        for genre, p in sorted(zip(GENRES, mean_probs_3s), key=lambda x: -x[1]):
            print(f"{genre:10s}: {p*100:5.2f}%")
    else:
        print("\n⚠️ Nenhum segmento de 3s extraído. Usando a previsão de 30s como base para o resultado combinado.")
        mean_probs_3s = probs_30s

    if os.path.exists(TMP_SPECTROGRAM):
        os.remove(TMP_SPECTROGRAM)


if __name__ == "__main__":
    main()