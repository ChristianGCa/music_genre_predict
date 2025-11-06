import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from glob import glob
import numpy as np

DATASET_PATH = "/home/christian/Documentos/music_dataset (Cópia)/Data/genres_original/"
OUTPUT_PATH  = "./spectrograms/"

sample_rate  = 22050
num_segments = 10  # 3 segundos
duration     = 30
samples_per_segment = int(sample_rate * duration / num_segments)

# Pastas de saída
full_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "full")
segments_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "segments")

os.makedirs(full_OUTPUT_PATH, exist_ok=True)
os.makedirs(segments_OUTPUT_PATH, exist_ok=True)

audio_files = glob(DATASET_PATH + "/*/*.wav")

for filepath in sorted(audio_files):
    genre = filepath.split("/")[-2]
    fname = filepath.split("/")[-1].replace(".wav", "")

    print(f"Processando {genre} - {fname}...")

    genre_full_dir = os.path.join(full_OUTPUT_PATH, genre)
    genre_seg_dir = os.path.join(segments_OUTPUT_PATH, genre)
    os.makedirs(genre_full_dir, exist_ok=True)
    os.makedirs(genre_seg_dir, exist_ok=True)

    try:
        y, sr = librosa.load(filepath, sr=sample_rate)
    except Exception as e:
        print("Erro ao carregar:", filepath, "|", e)
        continue

    # COMPLETO
    S_full = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=3072,
        hop_length=384,
        n_mels=192
    )
    S_full_dB = librosa.power_to_db(S_full, ref=np.max)

    plt.figure(figsize=(8,3))
    librosa.display.specshow(S_full_dB, sr=sr, cmap="magma")
    plt.axis('off')

    full_path = os.path.join(genre_full_dir, f"{fname}_full.png")
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # POR SEGMENTOS
    for n in range(num_segments):
        start = samples_per_segment * n
        end = start + samples_per_segment
        y_seg = y[start:end]

        S = librosa.feature.melspectrogram(
            y=y_seg,
            sr=sr,
            n_fft=3072,
            hop_length=384,
            n_mels=192
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(4,3))
        librosa.display.specshow(S_dB, sr=sr, cmap="magma")
        plt.axis('off')

        seg_path = os.path.join(genre_seg_dir, f"{fname}_{n}.png")
        plt.savefig(seg_path, bbox_inches='tight', pad_inches=0)
        plt.close()

print("\nEspectrogramas completos em:", full_OUTPUT_PATH)
print("Espectrogramas segmentados em:", segments_OUTPUT_PATH)
