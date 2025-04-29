import os
import numpy as np

# Configurações
PROCESSED_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/processed/'
NOISY_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/noisy/'
SEED = 42  # Para resultados reproduzíveis

def generate_noisy_signal(clean_signal, noise_level_range=(0, 16)):
    np.random.seed(SEED)
    noise = np.random.normal(0, 1, size=clean_signal.shape)  # Ruído branco gaussiano
    w_signal = 1
    w_noise = np.random.uniform(noise_level_range[0], noise_level_range[1])
    noisy_signal = w_signal * clean_signal + w_noise * noise
    return noisy_signal

if __name__ == "__main__":
    os.makedirs(NOISY_DIR, exist_ok=True)

    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npy')]
    print(f"🔍 {len(files)} segmentos encontrados para adicionar ruído.")

    for file_name in files:
        file_path = os.path.join(PROCESSED_DIR, file_name)
        clean_signal = np.load(file_path)

        noisy_signal = generate_noisy_signal(clean_signal)

        noisy_file_name = file_name.replace('.npy', '_noisy.npy')
        save_path = os.path.join(NOISY_DIR, noisy_file_name)

        np.save(save_path, noisy_signal)

    print(f"✅ Sinais ruidosos salvos em: {NOISY_DIR}")
