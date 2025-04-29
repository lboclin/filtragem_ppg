import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from gan_generator import PPGGenerator

# Configurações
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/generator.pth'
NOISY_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/noisy/'
OUTPUT_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/outputs/'

# Garante que a pasta de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    # Inicializa o Gerador
    generator = PPGGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()

    # Lista de sinais ruidosos
    noisy_files = [f for f in os.listdir(NOISY_DIR) if f.endswith('.npy')]

    print(f"🔍 {len(noisy_files)} arquivos encontrados para avaliação...")

    for file_name in noisy_files:
        # Carrega o sinal ruidoso
        noisy_signal = np.load(os.path.join(NOISY_DIR, file_name))
        noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # Gera o sinal limpo
        with torch.no_grad():
            filtered_signal = generator(noisy_tensor).cpu().squeeze().numpy()

        # Tempo
        t = np.linspace(0, len(noisy_signal) / 500, len(noisy_signal))  # Assumindo fs=500Hz

        # Plota o gráfico
        plt.figure(figsize=(12,6))
        plt.plot(t, noisy_signal, label='PPG Ruidoso')
        plt.plot(t, filtered_signal, label='PPG Filtrado (Gerador)', alpha=0.8)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Reconstrução do PPG - {file_name}')
        plt.legend()
        plt.grid()

        # Salva o gráfico na pasta outputs
        output_path = os.path.join(OUTPUT_DIR, file_name.replace('.npy', '.png'))
        plt.savefig(output_path)
        plt.close()

        print(f"✅ Gráfico salvo: {output_path}")

    print("🎯 Avaliação em lote concluída!")
