import librosa
from glob import glob
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Change, if necessary
DATASET_PATH = "/home/chris/Documents/music_dataset/Data/genres_original/"
OUTPUT_PATH  = "/home/chris/Documents/CSVs/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

num_mfcc = 20
sample_rate = 22050
n_fft = 2048
hop_length = 512
num_segment_second_block = 10
samples_per_segment_second_block = int(sample_rate * 30 / num_segment_second_block)

columns = [
    "filename", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
    "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo"
]
for x in range(1, 21):
    columns.append(f"mfcc{x}_mean")
    columns.append(f"mfcc{x}_var")
columns.append("label")

audio_files = sorted(glob(os.path.join(DATASET_PATH, "*/*")))
genres = sorted(set(os.path.basename(os.path.dirname(f)) for f in audio_files))
print("Genres", genres)

def extract_features(y, sr):
    feats = {}
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    feats["chroma_stft_mean"] = chroma.mean()
    feats["chroma_stft_var"] = chroma.var()

    rms = librosa.feature.rms(y=y)
    feats["rms_mean"] = rms.mean()
    feats["rms_var"] = rms.var()

    centroid = librosa.feature.spectral_centroid(y=y)
    feats["spectral_centroid_mean"] = centroid.mean()
    feats["spectral_centroid_var"] = centroid.var()

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats["spectral_bandwidth_mean"] = bandwidth.mean()
    feats["spectral_bandwidth_var"] = bandwidth.var()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats["rolloff_mean"] = rolloff.mean()
    feats["rolloff_var"] = rolloff.var()

    zcr = librosa.feature.zero_crossing_rate(y=y)
    feats["zero_crossing_rate_mean"] = zcr.mean()
    feats["zero_crossing_rate_var"] = zcr.var()

    harmony, perceptr = librosa.effects.hpss(y=y)
    feats["harmony_mean"] = harmony.mean()
    feats["harmony_var"] = harmony.var()
    feats["perceptr_mean"] = perceptr.mean()
    feats["perceptr_var"] = perceptr.var()

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = tempo

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length).T
    for x in range(20):
        feats[f"mfcc{x+1}_mean"] = mfcc[:, x].mean()
        feats[f"mfcc{x+1}_var"] = mfcc[:, x].var()

    return feats


def process_full_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        genre = os.path.basename(os.path.dirname(file_path))
        fname = os.path.basename(file_path)
        print(f"Processing full: {fname}")
        feats = extract_features(y, sr)
        feats["filename"] = fname
        feats["label"] = genre
        return feats
    except Exception as e:
        print(f"Process error {file_path}: {e}")
        return None


def process_segmented_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=sample_rate)
        genre = os.path.basename(os.path.dirname(file_path))
        fname = os.path.basename(file_path)
        print(f"Processing segments: {fname}")
        seg_features = []

        for n in range(num_segment_second_block):
            start = samples_per_segment_second_block * n
            end = samples_per_segment_second_block * (n + 1)
            y_seg = y[start:end]
            if len(y_seg) == 0:
                continue
            feats = extract_features(y_seg, sr)
            feats["filename"] = f"{os.path.splitext(fname)[0]}_{n}.wav"
            feats["label"] = genre
            seg_features.append(feats)

        return seg_features

    except Exception as e:
        print(f"Error: {file_path}: {e}")
        return []

print("\nGenerating full audio CSV")
results_full = []
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(process_full_audio, f) for f in audio_files]
    for f in as_completed(futures):
        res = f.result()
        if res:
            results_full.append(res)

df_full = pd.DataFrame(results_full)
df_full.to_csv(os.path.join(OUTPUT_PATH, 'data.csv'), index=False)
print("Full CSV saved")

print("\nGenerating 3 seconds segments CSV")
all_segments = []
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(process_segmented_audio, f) for f in audio_files]
    for f in as_completed(futures):
        segs = f.result()
        if segs:
            all_segments.extend(segs)

df_segments = pd.DataFrame(all_segments)
df_segments.to_csv(os.path.join(OUTPUT_PATH, 'features_3_sec.csv'), index=False)
print(f"3 seconds CSV saved")
