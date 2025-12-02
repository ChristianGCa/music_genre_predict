import os
from glob import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from audio_utils import (
    convert_to_wav,
    load_audio,
    extract_middle_segment,
    split_segments,
    save_mel_spectrogram,
    SR,
)

DATA_PATH = "/home/chris/Documents/music_dataset/Data/genres_original/"
OUT_PATH_30S = "./data/spectograms_30s/"
OUT_PATH_3S = "./data/spectograms_3s/"
CHUNK_DURATION = 3  # segundos

def ensure_wav(file_path):
    if file_path.lower().endswith(".wav"):
        return file_path
    return convert_to_wav(file_path)

genres = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

for genre in genres:
    os.makedirs(os.path.join(OUT_PATH_30S, genre), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH_3S, genre), exist_ok=True)

for genre in genres:
    files = glob(os.path.join(DATA_PATH, genre, "*"))
    print(f"\nProcessando {len(files)} arquivos de {genre}...")

    for idx, file_path in enumerate(files, 1):
        try:
            wav_path = ensure_wav(file_path)
            y = load_audio(wav_path, sr=SR)
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            continue

        try:
            y_30s = extract_middle_segment(wav_path, duration=30, sr=SR)
            base_name = os.path.basename(wav_path).replace(".wav", "_30s.png")
            out_30s = os.path.join(OUT_PATH_30S, genre, base_name)
            save_mel_spectrogram(y_30s, sr=SR, out_path=out_30s)
        except Exception as e:
            print(f"Erro ao gerar espectrograma 30s ({file_path}): {e}")
            continue

        try:
            chunks = split_segments(y, sr=SR, segment_sec=CHUNK_DURATION)
            for i, chunk in enumerate(chunks):
                out_3s = os.path.join(
                    OUT_PATH_3S, genre,
                    os.path.basename(wav_path).replace(".wav", f"_{i}.png")
                )
                save_mel_spectrogram(chunk, sr=SR, out_path=out_3s)
        except Exception as e:
            print(f"Erro ao gerar espectrogramas 3s ({file_path}): {e}")

        print(f"{idx}/{len(files)} -> {os.path.basename(file_path)} processado.")

print("\nTodos os espectrogramas foram gerados (arquivos problemáticos foram pulados).")
