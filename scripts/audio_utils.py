from PIL import Image
from torchvision import transforms
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment

SR = 22050

def convert_to_wav(file_path):
    """Converte MP3 para WAV (mantém se já for WAV)."""
    if file_path.lower().endswith(".mp3"):
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_mp3(file_path)
        audio.export(wav_path, format="wav")
        return wav_path
    return file_path

def load_audio(file_path, sr=SR):
    """Carrega o áudio e retorna o array mono."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y

def extract_middle_segment(wav_path, duration=30, sr=SR):
    """Extrai o trecho central de `duration` segundos de um arquivo WAV."""
    y, _ = librosa.load(wav_path, sr=sr)
    total_samples = len(y)
    seg_samples = duration * sr
    start = max(0, total_samples // 2 - seg_samples // 2)
    end = start + seg_samples
    return y[start:end]

def split_segments(y, sr=SR, segment_sec=3):
    """Divide o sinal em segmentos fixos de N segundos."""
    seg_len = segment_sec * sr
    return [y[i:i+seg_len] for i in range(0, len(y), seg_len) if len(y[i:i+seg_len]) == seg_len]

def save_mel_spectrogram(y, sr=SR, out_path=None, n_mels=128, fmax=8000):
    """
    Gera espectrograma mel e salva como imagem PNG.
    Se `out_path` for None, apenas retorna o array em dB.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if out_path:
        plt.figure(figsize=(3, 3))
        plt.axis('off')
        librosa.display.specshow(S_dB, sr=sr, fmax=fmax)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return S_dB

def extract_features(y, sr=SR):
    """Extrai features estatísticas de áudio."""
    feats = {}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    feats.update({
        'chroma_mean': chroma.mean(), 'chroma_std': chroma.std(),
        'rms_mean': rms.mean(), 'rms_std': rms.std(),
        'spec_cent_mean': spec_cent.mean(), 'spec_cent_std': spec_cent.std(),
        'spec_bw_mean': spec_bw.mean(), 'spec_bw_std': spec_bw.std(),
        'rolloff_mean': rolloff.mean(), 'rolloff_std': rolloff.std(),
        'zcr_mean': zcr.mean(), 'zcr_std': zcr.std(),
    })

    for i in range(20):
        feats[f'mfcc{i+1}_mean'] = mfccs[i].mean()
        feats[f'mfcc{i+1}_std'] = mfccs[i].std()

    return feats


def load_spectrogram_as_tensor(image_path, img_size=(128, 128)):
    """
    Carrega uma imagem de espectrograma e converte em tensor normalizado
    compatível com a CNN usada no projeto.

    Args:
        image_path (str): Caminho da imagem PNG gerada pelo mel spectrogram.
        img_size (tuple): Tamanho alvo (largura, altura).

    Returns:
        torch.Tensor: Tensor 4D no formato (1, 1, H, W).
    """
    img = Image.open(image_path).convert("L")  # converte para escala de cinza
    img = img.resize(img_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # mesma normalização usada no treino
    ])

    tensor = transform(img).unsqueeze(0)  # adiciona dimensão batch
    return tensor
