import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from audio_utils import (
    convert_to_wav,
    extract_middle_segment,
    extract_features,
    split_segments,
    SR
)

from train_models.train_model_csvs import MLP  # mesma arquitetura usada no treino


def load_label_encoder(label_path):
    classes = np.load(label_path, allow_pickle=True)
    return classes


def extract_features_tensor(audio_data):
    feats = extract_features(audio_data, SR)
    values = np.array(list(feats.values()), dtype=np.float32)
    return torch.tensor(values).unsqueeze(0), feats


def predict_genres_mlp(audio_path, model_30s_dir, model_3s_dir, out_root="./predictions_mlp"):

    # ======================
    # 1) Converter para WAV
    # ======================
    wav_path = convert_to_wav(audio_path)

    # ======================
    # 2) Extrair segmento completo de 30s
    # ======================
    y = extract_middle_segment(wav_path, duration=30.0, sr=SR)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    out_dir = os.path.join(out_root, f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # ======================
    # 3) Carregar modelos & label encoders
    # ======================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes_30s = load_label_encoder(os.path.join(model_30s_dir, "label_encoder.npy"))
    classes_3s = load_label_encoder(os.path.join(model_3s_dir, "label_encoder.npy"))

    model_30s = MLP(input_dim=None, num_classes=len(classes_30s))  # ajustado ao carregar pesos
    model_3s = MLP(input_dim=None, num_classes=len(classes_3s))

    # Carrega pesos
    weights_30s = torch.load(os.path.join(model_30s_dir, "model.pth"), map_location=device)
    weights_3s = torch.load(os.path.join(model_3s_dir, "model.pth"), map_location=device)

    # Detecta input dim automaticamente
    first_key = list(weights_30s.keys())[0]
    input_dim_30s = weights_30s[first_key].shape[1]
    input_dim_3s = weights_3s[first_key].shape[1]

    model_30s = MLP(input_dim_30s, len(classes_30s)).to(device)
    model_3s = MLP(input_dim_3s, len(classes_3s)).to(device)

    model_30s.load_state_dict(weights_30s)
    model_3s.load_state_dict(weights_3s)

    model_30s.eval()
    model_3s.eval()

    # ======================
    # 4) Extrair features 30s
    # ======================
    tensor_30s, feats_30s_dict = extract_features_tensor(y)

    # ======================
    # 5) Extrair features para segmentos 3s
    # ======================
    segments = split_segments(y, sr=SR, segment_sec=3)
    tensors_3s = []
    feats_3s_dict_list = []

    for seg in segments:
        t, d = extract_features_tensor(seg)
        tensors_3s.append(t)
        feats_3s_dict_list.append(d)

    # ======================
    # 6) Predições
    # ======================
    with torch.no_grad():
        probs_30s = torch.softmax(model_30s(tensor_30s.to(device)), dim=1).cpu().numpy().flatten()

        if len(tensors_3s) > 0:
            stacked = torch.cat(tensors_3s).to(device)
            probs_3s_all = torch.softmax(model_3s(stacked), dim=1).cpu().numpy()
            probs_3s_mean = probs_3s_all.mean(axis=0)
        else:
            probs_3s_mean = None

    # ======================
    # 7) Ordenação e exibição
    # ======================
    sorted_30s = sorted(zip(classes_30s, probs_30s), key=lambda x: x[1], reverse=True)

    print("\n🎵 Predição usando FEATURES 30s:\n")
    for genre, prob in sorted_30s:
        print(f"{genre}: {prob*100:.2f}%")

    if probs_3s_mean is not None:
        sorted_3s = sorted(zip(classes_3s, probs_3s_mean), key=lambda x: x[1], reverse=True)
        print("\n🎵 Predição usando FEATURES 3s (média dos segmentos):\n")
        for genre, prob in sorted_3s:
            print(f"{genre}: {prob*100:.2f}%")
    else:
        sorted_3s = None
        print("\nNenhum segmento de 3s disponível.\n")

    # ======================
    # 8) Salvar metadata
    # ======================
    metadata = {
        "audio_path": audio_path,
        "wav_used": wav_path,
        "timestamp": ts,
        "device": str(device),
        "model_30s_dir": model_30s_dir,
        "model_3s_dir": model_3s_dir,
        "genres_30s": list(classes_30s),
        "probabilities_30s": {g: float(p) for g, p in zip(classes_30s, probs_30s)},
        "sorted_30s": [(str(g), float(p)) for g, p in sorted_30s],
        "genres_3s": list(classes_3s),
        "probabilities_3s_mean": {g: float(p) for g, p in zip(classes_3s, probs_3s_mean)} if probs_3s_mean is not None else None,
        "sorted_3s": [(str(g), float(p)) for g, p in sorted_3s] if sorted_3s is not None else None,
        "n_segments_3s": len(segments)
    }

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Resultados salvos em:\n{out_dir}\n")

    return probs_30s, probs_3s_mean, out_dir


# ===== EXECUÇÃO DIRETA =====
if __name__ == "__main__":
    audio_path = "/home/christian/Músicas/music2.mp3"

    model_30s_dir = "./models/mlp_30s"
    model_3s_dir = "./models/mlp_3s"

    predict_genres_mlp(audio_path, model_30s_dir, model_3s_dir)
