from PIL import Image
from torchvision import transforms
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import shutil

SR = 22050

def convert_to_wav(file_path, out_dir='converted_wavs'):
    """Converte MP3 para WAV e guarda o WAV numa pasta auxiliar dentro do projeto."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0] + '.wav'
        out_path = os.path.join(out_dir, base_name)

        if file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            audio.export(out_path, format='wav')
            print(f"Converted MP3 -> WAV: {out_path}")
            return out_path

        if file_path.lower().endswith('.wav'):
            try:
                src_abs = os.path.abspath(file_path)
                dst_abs = os.path.abspath(out_path)
                if src_abs != dst_abs:
                    shutil.copy(file_path, out_path)
                    print(f"[INFO] Copied WAV to auxiliary folder: {out_path}")
            except Exception:
                return file_path
            return out_path

        try:
            audio = AudioSegment.from_file(file_path)
            audio.export(out_path, format='wav')
            print(f"[INFO] Converted file -> WAV: {out_path}")
            return out_path
        except Exception:
            return file_path
    except Exception as e:
        print(f"Falha ao criar pasta de conversão ou converter arquivo: {e}")
        return file_path

def load_audio(file_path, sr=SR):
    """Carrega o áudio e retorna o array mono."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    return y

def extract_middle_segment(wav_path, duration=3.0, sr=22050):
    try:
        y, _ = librosa.load(wav_path, sr=sr)
    except Exception as e:
        print(f"[AVISO] Erro ao carregar {wav_path}: {e}")
        return np.zeros(int(sr * duration))

    total_duration = librosa.get_duration(y=y, sr=sr)
    if total_duration < duration:
        return np.pad(y, (0, int(sr * duration) - len(y)), mode="constant")

    start = int((total_duration / 2 - duration / 2) * sr)
    end = start + int(duration * sr)
    return y[start:end]

def split_segments(y, sr=SR, segment_sec=3):
    """Divide o sinal em segmentos fixos de N segundos."""
    seg_len = segment_sec * sr
    return [y[i:i+seg_len] for i in range(0, len(y), seg_len) if len(y[i:i+seg_len]) == seg_len]

def save_mel_spectrogram(y, sr=SR, out_path=None, n_mels=256, fmax=10000, img_size=256, n_fft=1024, hop_length=256):
    """
    Gera espectrograma mel e salva como imagem PNG em alta resolução, com alta definição temporo-frequencial.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa.display
    import numpy as np

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    if out_path:
        fig = plt.figure(figsize=(img_size / 100, img_size / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        librosa.display.specshow(S_dB, sr=sr, fmax=fmax, ax=ax, hop_length=hop_length)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

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


def load_spectrogram_as_tensor(image_path, img_size=(512, 512)):
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
        transforms.Normalize([0.5], [0.5])
    ])

    tensor = transform(img).unsqueeze(0)
    return tensor
