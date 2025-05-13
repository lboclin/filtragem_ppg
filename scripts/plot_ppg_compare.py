import os
import numpy as np
import matplotlib.pyplot as plt

# Pastas de entrada e saída
PROCESSED_DIR = r"C:\Users\Usuário\Desktop\codes\projeto caes\filtragem_ppg\data\processed"
NOISY_DIR = r"C:\Users\Usuário\Desktop\codes\projeto caes\filtragem_ppg\data\noisy_realistic"
OUTPUT_DIR = r"C:\Users\Usuário\Desktop\codes\projeto caes\filtragem_ppg\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lista de arquivos ruidosos com sufixo "_realistic.npy"
noisy_files = [f for f in os.listdir(NOISY_DIR) if f.endswith("_realistic.npy")]

print(f"🔍 Gerando gráficos comparativos para {len(noisy_files)} segmentos...")

for noisy_file in noisy_files:
    clean_file = noisy_file.replace("_realistic", "")
    clean_path = os.path.join(PROCESSED_DIR, clean_file)
    noisy_path = os.path.join(NOISY_DIR, noisy_file)

    # Verifica se o correspondente limpo existe
    if not os.path.exists(clean_path):
        print(f"⚠️ Arquivo limpo não encontrado para: {noisy_file}")
        continue

    # Carrega os dados
    clean = np.load(clean_path)
    noisy = np.load(noisy_path)

    # Cria o gráfico
    plt.figure(figsize=(12, 4))
    plt.plot(clean[:1500], label="PPG Limpo", linewidth=1.5)
    plt.plot(noisy[:1500], label="PPG com Ruído Realista", linewidth=1.2, alpha=0.75)
    plt.title(f"Comparação: {clean_file}")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Salva o gráfico
    output_file = os.path.join(OUTPUT_DIR, clean_file.replace(".npy", "_compare.png"))
    plt.savefig(output_file)
    plt.close()

print(f"✅ Gráficos salvos em: {OUTPUT_DIR}")
