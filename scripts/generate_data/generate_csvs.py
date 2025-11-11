import os
import sys
from glob import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Ajusta o path para encontrar o módulo audio_utils ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from audio_utils import (
    convert_to_wav,
    load_audio,
    extract_middle_segment,
    extract_features,
    split_segments,
    SR,
)

# --- Configurações ---
DATA_DIR = "/home/christian/Documentos/music_dataset/Data/genres_original/"
CSV_30S = "./data/csv/features_30s.csv"
CSV_3S = "./data/csv/features_3s.csv"
SEGMENT_DURATION = 3   # segundos
DURATION_30S = 30      # segundos

os.makedirs("./data/csv", exist_ok=True)

# --- Carrega todos os arquivos (WAV ou MP3) ---
all_files = []
for genre_folder in os.listdir(DATA_DIR):
    genre_path = os.path.join(DATA_DIR, genre_folder)
    if os.path.isdir(genre_path):
        audio_files = glob(os.path.join(genre_path, "*"))
        all_files.extend([(f, genre_folder) for f in audio_files])

print(f"🎵 Encontrados {len(all_files)} arquivos de áudio em {DATA_DIR}\n")

# =====================================================
# 1️⃣ CSV DE 30 SEGUNDOS (trecho central)
# =====================================================
rows_30s = []
for idx, (file_path, genre) in enumerate(all_files, 1):
    try:
        wav_path = convert_to_wav(file_path)
        y_30s = extract_middle_segment(wav_path, duration=DURATION_30S, sr=SR)

        if len(y_30s) < DURATION_30S * SR:
            print(f"⚠️ Pulando {file_path}: duração menor que {DURATION_30S}s.")
            continue

        feats = extract_features(y_30s, SR)
        feats["filename"] = os.path.basename(file_path)
        feats["genre"] = genre
        rows_30s.append(feats)

        print(f"[{idx}/{len(all_files)}] ✅ 30s: {file_path}")
    except Exception as e:
        print(f"❌ Erro em {file_path}: {e}")

# --- Salva CSV 30s ---
if rows_30s:
    df_30s = pd.DataFrame(rows_30s)
    feature_cols = [c for c in df_30s.columns if c not in ["filename", "genre"]]
    scaler = StandardScaler()
    df_30s[feature_cols] = scaler.fit_transform(df_30s[feature_cols])
    df_30s.to_csv(CSV_30S, index=False)
    print(f"\n💾 CSV de 30s salvo em: {CSV_30S}")
else:
    print("\n⚠️ Nenhum arquivo válido gerado para CSV de 30 segundos.")

# =====================================================
# 2️⃣ CSV DE 3 SEGUNDOS (segmentos do trecho central)
# =====================================================
rows_3s = []
for idx, (file_path, genre) in enumerate(all_files, 1):
    try:
        wav_path = convert_to_wav(file_path)
        y_30s = extract_middle_segment(wav_path, duration=DURATION_30S, sr=SR)

        if len(y_30s) < DURATION_30S * SR:
            print(f"⚠️ Pulando {file_path}: duração menor que {DURATION_30S}s.")
            continue

        segments = split_segments(y_30s, sr=SR, segment_sec=SEGMENT_DURATION)
        for s_idx, seg in enumerate(segments, 1):
            feats = extract_features(seg, SR)
            feats["filename"] = os.path.basename(file_path)
            feats["segment"] = s_idx
            feats["genre"] = genre
            rows_3s.append(feats)

        print(f"[{idx}/{len(all_files)}] ✅ 3s: {file_path} ({len(segments)} segmentos)")
    except Exception as e:
        print(f"❌ Erro em {file_path}: {e}")

# --- Salva CSV 3s ---
if rows_3s:
    df_3s = pd.DataFrame(rows_3s)
    feature_cols_3s = [c for c in df_3s.columns if c not in ["filename", "segment", "genre"]]
    scaler_3s = StandardScaler()
    df_3s[feature_cols_3s] = scaler_3s.fit_transform(df_3s[feature_cols_3s])
    df_3s.to_csv(CSV_3S, index=False)
    print(f"\n💾 CSV de 3s salvo em: {CSV_3S}")
else:
    print("\n⚠️ Nenhum arquivo válido gerado para CSV de 3 segundos.")

print("\n✅ Extração concluída com sucesso!")
