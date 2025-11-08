import os
import librosa
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler

# --- Configurações ---
DATA_DIR = "/home/christian/Documentos/music_dataset/Data/genres_original/"
CSV_30S = "./data/csv/features_30s.csv"
CSV_3S = "./data/csv/features_3s.csv"
SR = 22050
SEGMENT_DURATION = 3  # segundos
DURATION_30S = 30  # segundos

os.makedirs("./data/csv", exist_ok=True)

def extract_features(y, sr):
    features = {}
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = chroma.mean()
    features['chroma_std'] = chroma.std()
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = rms.mean()
    features['rms_std'] = rms.std()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_cent_mean'] = spec_cent.mean()
    features['spec_cent_std'] = spec_cent.std()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spec_bw_mean'] = spec_bw.mean()
    features['spec_bw_std'] = spec_bw.std()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['rolloff_mean'] = rolloff.mean()
    features['rolloff_std'] = rolloff.std()
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = zcr.mean()
    features['zcr_std'] = zcr.std()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}_mean'] = mfccs[i].mean()
        features[f'mfcc{i+1}_std'] = mfccs[i].std()
    return features

def split_segments(y, sr, segment_sec):
    seg_len = segment_sec * sr
    segments = []
    for start in range(0, len(y), seg_len):
        seg = y[start:start+seg_len]
        if len(seg) == seg_len:
            segments.append(seg)
    return segments

all_files = []
for genre_folder in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, genre_folder)
    if os.path.isdir(path):
        files = glob(os.path.join(path, "*.wav"))
        all_files.extend([(f, genre_folder) for f in files])

rows_30s = []
print("Gerando CSV de 30 segundos...")
for idx, (file_path, genre) in enumerate(all_files, 1):
    try:
        y, _ = librosa.load(file_path, sr=SR)
        if len(y) > DURATION_30S * SR:
            start = len(y)//2 - (DURATION_30S*SR)//2
            y = y[start:start + DURATION_30S*SR]
        feats = extract_features(y, SR)
        feats['filename'] = os.path.basename(file_path)
        feats['genre'] = genre
        rows_30s.append(feats)
        print(f"Processado {idx}/{len(all_files)}: {file_path}")
    except Exception as e:
        print(f"Erro em {file_path}, pulando. Erro: {e}")

df_30s = pd.DataFrame(rows_30s)

feature_cols = [c for c in df_30s.columns if c not in ['filename','genre']]
scaler = StandardScaler()
df_30s[feature_cols] = scaler.fit_transform(df_30s[feature_cols])

df_30s.to_csv(CSV_30S, index=False)
print(f"CSV de 30s salvo em: {CSV_30S}")

rows_3s = []
print("Gerando CSV de 3 segundos...")
for idx, (file_path, genre) in enumerate(all_files, 1):
    try:
        y, _ = librosa.load(file_path, sr=SR)
        if len(y) > DURATION_30S * SR:
            start = len(y)//2 - (DURATION_30S*SR)//2
            y = y[start:start + DURATION_30S*SR]
        segments = split_segments(y, SR, SEGMENT_DURATION)
        for s_idx, seg in enumerate(segments,1):
            feats = extract_features(seg, SR)
            feats['filename'] = os.path.basename(file_path)
            feats['segment'] = s_idx
            feats['genre'] = genre
            rows_3s.append(feats)
        print(f"Processado {idx}/{len(all_files)}: {file_path} ({len(segments)} segmentos)")
    except Exception as e:
        print(f"Erro em {file_path}, pulando. Erro: {e}")

df_3s = pd.DataFrame(rows_3s)
feature_cols_3s = [c for c in df_3s.columns if c not in ['filename','segment','genre']]
scaler_3s = StandardScaler()
df_3s[feature_cols_3s] = scaler_3s.fit_transform(df_3s[feature_cols_3s])

df_3s.to_csv(CSV_3S, index=False)
print(f"CSV de 3s salvo em: {CSV_3S}")
