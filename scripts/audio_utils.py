import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def load_audio(wav_path, sr=22050):
    """Carrega arquivo WAV e retorna array mono."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    return y

def split_audio(y, sr=22050, chunk_duration=3):
    """Divide o áudio em pedaços de chunk_duration segundos."""
    chunk_size = chunk_duration * sr
    chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
    if len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]
    return chunks

def save_mel_spectrogram(y, sr, out_path, n_mels=128, fmax=8000):
    """Gera espectrograma mel e salva como PNG."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3,3))
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, fmax=fmax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
