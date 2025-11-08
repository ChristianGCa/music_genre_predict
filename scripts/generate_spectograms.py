import os
from glob import glob
from audio_utils import load_audio, split_audio, save_mel_spectrogram

DATA_PATH = "/home/christian/Documentos/music_dataset/Data/genres_original/"
OUT_PATH_30S = "./data/spectrograms_30s/"
OUT_PATH_3S = "./data/spectrograms_3s/"

CHUNK_DURATION = 3  # segundos
SR = 22050

genres = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

for genre in genres:
    os.makedirs(os.path.join(OUT_PATH_30S, genre), exist_ok=True)
    os.makedirs(os.path.join(OUT_PATH_3S, genre), exist_ok=True)

for genre in genres:
    wav_files = glob(os.path.join(DATA_PATH, genre, "*.wav"))
    print(f"Processando {len(wav_files)} arquivos de {genre}...")

    for idx, wav_file in enumerate(wav_files, 1):
        try:
            y = load_audio(wav_file, sr=SR)
        except Exception as e:
            print(f"⚠️  Não foi possível ler o arquivo: {wav_file}. Pulando. Erro: {e}")
            continue

        y_30s = y[:30*SR]  # pega até 30s
        base_name = os.path.basename(wav_file).replace(".wav", "_30s.png")
        out_file_30s = os.path.join(OUT_PATH_30S, genre, base_name)
        try:
            save_mel_spectrogram(y_30s, sr=SR, out_path=out_file_30s)
        except Exception as e:
            print(f"⚠️  Erro ao gerar espectrograma 30s: {wav_file}. Pulando. Erro: {e}")
            continue

        chunks = split_audio(y, sr=SR, chunk_duration=CHUNK_DURATION)
        for i, chunk in enumerate(chunks):
            base_name_chunk = os.path.basename(wav_file).replace(".wav", f"_{i}.png")
            out_file_3s = os.path.join(OUT_PATH_3S, genre, base_name_chunk)
            try:
                save_mel_spectrogram(chunk, sr=SR, out_path=out_file_3s)
            except Exception as e:
                print(f"⚠️  Erro ao gerar espectrograma 3s: {wav_file}, chunk {i}. Pulando. Erro: {e}")

        print(f"Processado: {wav_file}  /  {len(wav_files)}")

print("Todos os espectrogramas foram gerados (arquivos problemáticos foram pulados)!")
