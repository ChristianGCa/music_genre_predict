import os
import sys
import torch
import numpy as np
import shutil
import json
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from audio_utils import convert_to_wav, extract_middle_segment, save_mel_spectrogram, load_spectrogram_as_tensor
from train_models.train_model_spectograms import SpectrogramCNN, GENRES, IMG_SIZE

import tempfile

def predict_genres(audio_path, model_30s_path, model_3s_path, out_root="./predictions"):
    wav_path = convert_to_wav(audio_path)

    y = extract_middle_segment(wav_path, duration=30.0)

    os.makedirs(out_root, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(out_root, f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_30s_path = os.path.join(tmpdir, 'spec_30s.png')
        save_mel_spectrogram(y, out_path=spec_30s_path, img_size=IMG_SIZE[0])
        tensor_30s = load_spectrogram_as_tensor(spec_30s_path, img_size=IMG_SIZE)
        try:
            shutil.copy(spec_30s_path, os.path.join(out_dir, 'spec_30s.png'))
        except Exception:
            pass

        from audio_utils import split_segments
        segments = split_segments(y, segment_sec=3)
        tensors_3s = []
        for i, seg in enumerate(segments):
            seg_path = os.path.join(tmpdir, f'spec_3s_{i}.png')
            save_mel_spectrogram(seg, out_path=seg_path, img_size=IMG_SIZE[0])
            tensors_3s.append(load_spectrogram_as_tensor(seg_path, img_size=IMG_SIZE))
            try:
                shutil.copy(seg_path, os.path.join(out_dir, f'spec_3s_{i}.png'))
            except Exception:
                pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_30s = SpectrogramCNN(num_classes=len(GENRES)).to(device)
    model_30s.load_state_dict(torch.load(model_30s_path, map_location=device))
    model_30s.eval()

    model_3s = SpectrogramCNN(num_classes=len(GENRES)).to(device)
    model_3s.load_state_dict(torch.load(model_3s_path, map_location=device))
    model_3s.eval()

    with torch.no_grad():
        out_30s = model_30s(tensor_30s.to(device))
        probs_30s = torch.softmax(out_30s, dim=1).cpu().numpy().flatten()

        if len(tensors_3s) > 0:
            tensor_3s = torch.cat(tensors_3s, dim=0)
            out_3s = model_3s(tensor_3s.to(device))
            probs_3s = torch.softmax(out_3s, dim=1).cpu().numpy()
            probs_3s_mean = probs_3s.mean(axis=0)
        else:
            probs_3s_mean = None

    sorted_30s = sorted(zip(GENRES, probs_30s), key=lambda x: x[1], reverse=True)
    if probs_3s_mean is not None:
        sorted_3s = sorted(zip(GENRES, probs_3s_mean), key=lambda x: x[1], reverse=True)

    print('Predição usando espectrograma de 30s:\n')
    for genre, prob in sorted_30s:
        print(f'{genre}: {prob*100:.2f}%')

    print('\nPredição usando segmentos de 3s:\n')
    if probs_3s_mean is not None:
        for genre, prob in sorted_3s:
            print(f'{genre}: {prob*100:.2f}%')
    else:
        print('Nenhum segmento de 3s encontrado para prever.')

    top30_serial = [(str(g), float(p)) for g, p in sorted_30s]
    top3_serial = None
    if probs_3s_mean is not None:
        top3_serial = [(str(g), float(p)) for g, p in sorted_3s]

    metadata = {
        'audio_path': audio_path,
        'wav_used': os.path.basename(wav_path),
        'model_30s': model_30s_path,
        'model_3s': model_3s_path,
        'device': str(device),
        'timestamp': ts,
        'genres': list(GENRES),
        'probabilities_30s': {str(g): float(p) for g, p in zip(GENRES, probs_30s)},
        'probabilities_3s_mean': {str(g): float(p) for g, p in zip(GENRES, probs_3s_mean)} if probs_3s_mean is not None else None,
        'top_30s': top30_serial,
        'top_3s': top3_serial,
        'n_3s_segments': int(len(tensors_3s))
    }

    metadata_path = os.path.join(out_dir, 'metadata.json')
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'Falha ao salvar metadata.json: {e}')

    print(f"Resultados salvos em: {out_dir}")

    return probs_30s, probs_3s_mean, out_dir


if __name__ == '__main__':

    audio_path = '/home/christian/Músicas/music6.mp3'
    model_30s_path = 'models/cnn_30s.pth'
    model_3s_path = 'models/cnn_3s.pth'

    predict_genres(audio_path, model_30s_path, model_3s_path)
