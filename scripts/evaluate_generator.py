import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from gan_generator import PPGGenerator


# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/generator.pth'
NOISY_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/noisy/'

if __name__ == "__main__":
    # Inicializa o Gerador
    generator = PPGGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()

    # Escolhe um exemplo ruidoso para testar
    noisy_files = [f for f in os.listdir(NOISY_DIR) if f.endswith('.npy')]
    sample_file = np.random.choice(noisy_files)

    # Carrega o sinal ruidoso
    noisy_signal = np.load(os.path.join(NOISY_DIR, sample_file))
    noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Usa o Gerador para filtrar
    with torch.no_grad():
        filtered_signal = generator(noisy_tensor).cpu().squeeze().numpy()

    # Plota resultado
    t = np.linspace(0, len(noisy_signal) / 500, len(noisy_signal))

    plt.figure(figsize=(12,6))
    plt.plot(t, noisy_signal, label='PPG Ruidoso')
    plt.plot(t, filtered_signal, label='PPG Filtrado (Gerador)', alpha=0.8)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Reconstrução do PPG - {sample_file}')
    plt.legend()
    plt.grid()
    plt.show()
