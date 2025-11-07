# This script generates Mel-spectrograms from audio files, for both full tracks and 3-second segments.
# It saves these spectrograms as PNG images in specified output directories.

import os
import librosa
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor, as_completed

# CHANGE, IF NECESSARY
DATASET_PATH = "/home/chris/Documents/music_dataset/Data/genres_original/"
OUTPUT_PATH  = "/home/chris/Documents/spectrograms/"

sample_rate  = 22050
num_segments = 10  # 3 seconds
duration     = 30
samples_per_segment = int(sample_rate * duration / num_segments)

# Parameters tuned for lower-resolution images suitable for CNN training
TARGET_SIZE = (224, 224)  # (width, height) in pixels
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Output directories
full_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "full")
segments_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "segments")

os.makedirs(full_OUTPUT_PATH, exist_ok=True)
os.makedirs(segments_OUTPUT_PATH, exist_ok=True)

audio_files = glob(DATASET_PATH + "/*/*.wav")

def process_file(filepath):
    genre = filepath.split("/")[-2]
    fname = filepath.split("/")[-1].replace(".wav", "")

    print(f"Processing: {genre} - {fname}...")

    genre_full_dir = os.path.join(full_OUTPUT_PATH, genre)
    genre_seg_dir = os.path.join(segments_OUTPUT_PATH, genre)
    os.makedirs(genre_full_dir, exist_ok=True)
    os.makedirs(genre_seg_dir, exist_ok=True)

    try:
        y, sr = librosa.load(filepath, sr=sample_rate)
    except Exception as e:
        print("Erro ao carregar:", filepath, "|", e)
        return

    # FULL
    S_full = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    S_full_dB = librosa.power_to_db(S_full, ref=np.max)
    norm = (S_full_dB - S_full_dB.min()) / (S_full_dB.max() - S_full_dB.min() + 1e-6)
    norm = np.flipud(norm)
    rgba = cm.get_cmap('magma')(norm)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    img = img.resize(TARGET_SIZE, Image.LANCZOS)
    full_path = os.path.join(genre_full_dir, f"{fname}_full.png")
    img.save(full_path)

    # PER SEGMENTS
    for n in range(num_segments):
        start = samples_per_segment * n
        end = start + samples_per_segment
        y_seg = y[start:end]

        S = librosa.feature.melspectrogram(
            y=y_seg,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        norm_s = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-6)
        norm_s = np.flipud(norm_s)
        rgba_s = cm.get_cmap('magma')(norm_s)
        rgb_s = (rgba_s[:, :, :3] * 255).astype(np.uint8)
        img_s = Image.fromarray(rgb_s)
        img_s = img_s.resize(TARGET_SIZE, Image.LANCZOS)
        seg_path = os.path.join(genre_seg_dir, f"{fname}_{n}.png")
        img_s.save(seg_path)

# Parallel processing of audio files
max_workers = min(4, os.cpu_count() or 1)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_file, fp) for fp in sorted(audio_files)]
    for future in as_completed(futures):
        pass  # Awaits all threads to finish

print("\nSpectograms full: ", full_OUTPUT_PATH)
print("Spectograms segmenteds: ", segments_OUTPUT_PATH)
