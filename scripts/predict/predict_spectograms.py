import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from audio_utils import convert_to_wav, extract_middle_segment, save_mel_spectrogram, load_spectrogram_as_tensor
from train_models.train_model_spectograms import ImprovedCNN, GENRES, IMG_SIZE

import tempfile

def predict_genres(audio_path, model_30s_path, model_3s_path):
    # 1. Converter para wav
    wav_path = convert_to_wav(audio_path)

    # 2. Extrair 30s do meio
    y = extract_middle_segment(wav_path, duration=30.0)

    # 3. Gerar espectrograma 30s
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_30s_path = os.path.join(tmpdir, 'spec_30s.png')
        # Use the same IMG_SIZE used during training to ensure consistent preprocessing
        save_mel_spectrogram(y, out_path=spec_30s_path, img_size=IMG_SIZE[0])
        tensor_30s = load_spectrogram_as_tensor(spec_30s_path, img_size=IMG_SIZE)

        # 4. Gerar espectrogramas 3s
        from audio_utils import split_segments
        segments = split_segments(y, segment_sec=3)
        tensors_3s = []
        for i, seg in enumerate(segments):
            seg_path = os.path.join(tmpdir, f'spec_3s_{i}.png')
            save_mel_spectrogram(seg, out_path=seg_path, img_size=IMG_SIZE[0])
            tensors_3s.append(load_spectrogram_as_tensor(seg_path, img_size=IMG_SIZE))

    # 5. Carregar modelos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_30s = ImprovedCNN(num_classes=len(GENRES)).to(device)
    model_30s.load_state_dict(torch.load(model_30s_path, map_location=device))
    model_30s.eval()

    model_3s = ImprovedCNN(num_classes=len(GENRES)).to(device)
    model_3s.load_state_dict(torch.load(model_3s_path, map_location=device))
    model_3s.eval()

    # 6. Predizer 30s
    with torch.no_grad():
        out_30s = model_30s(tensor_30s.to(device))
        probs_30s = torch.softmax(out_30s, dim=1).cpu().numpy().flatten()

        # 7. Predizer 3s (média das probabilidades)
        if len(tensors_3s) > 0:
            tensor_3s = torch.cat(tensors_3s, dim=0)
            out_3s = model_3s(tensor_3s.to(device))
            probs_3s = torch.softmax(out_3s, dim=1).cpu().numpy()
            probs_3s_mean = probs_3s.mean(axis=0)
        else:
            probs_3s_mean = None

    # ---------------------------
    # ORDENAR RESULTADOS DO MAIOR PARA O MENOR
    # ---------------------------
    sorted_30s = sorted(zip(GENRES, probs_30s), key=lambda x: x[1], reverse=True)
    if probs_3s_mean is not None:
        sorted_3s = sorted(zip(GENRES, probs_3s_mean), key=lambda x: x[1], reverse=True)

    # 8. Mostrar resultados
    print('Predição usando espectrograma de 30s:')
    for genre, prob in sorted_30s:
        print(f'{genre}: {prob*100:.2f}%')

    print('\nPredição usando segmentos de 3s:')
    if probs_3s_mean is not None:
        for genre, prob in sorted_3s:
            print(f'{genre}: {prob*100:.2f}%')
    else:
        print('Nenhum segmento de 3s encontrado para prever.')

    return probs_30s, probs_3s_mean


if __name__ == '__main__':
    audio_path = '/home/christian/Músicas/music1.mp3'
    model_30s_path = 'models/cnn_30s.pth'
    model_3s_path = 'models/cnn_3s.pth'
    predict_genres(audio_path, model_30s_path, model_3s_path)
