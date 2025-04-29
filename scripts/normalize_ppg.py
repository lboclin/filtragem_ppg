import os
import numpy as np

# Configurações
PROCESSED_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/processed/'
NOISY_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/noisy/'

def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized = 2 * (signal - min_val) / (max_val - min_val) - 1  # Normaliza para [-1, 1]
    return normalized

def normalize_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    print(f"🔍 Normalizando {len(files)} arquivos em: {directory}")

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        signal = np.load(file_path)
        signal_norm = normalize_signal(signal)
        np.save(file_path, signal_norm)

if __name__ == "__main__":
    normalize_directory(PROCESSED_DIR)
    normalize_directory(NOISY_DIR)
    print("✅ Normalização concluída!")
