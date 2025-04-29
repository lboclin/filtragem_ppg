import os
import numpy as np
import wfdb

# Configuração
SAVE_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/raw/'
OUTPUT_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/processed/'

def load_ppg_signal(record_name):
    record_path = os.path.join(SAVE_DIR, record_name)
    record = wfdb.rdrecord(record_path)

    # Pega o primeiro canal "PLETH"
    ppg_index = None
    for idx, signal_name in enumerate(record.sig_name):
        if 'PLETH' in signal_name.upper():
            ppg_index = idx
            break

    if ppg_index is None:
        raise ValueError(f"Canal PPG não encontrado no registro {record_name}")
    
    ppg_signal = record.p_signal[:, ppg_index]
    return ppg_signal, record.fs

def segment_signal(signal, fs, window_sec=30):
    window_size = fs * window_sec  # Número de amostras por janela
    segments = []
    total_samples = len(signal)
    
    for start in range(0, total_samples, window_size):
        end = start + window_size
        if end <= total_samples:
            segment = signal[start:end]
            segments.append(segment)
    
    return np.array(segments)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    record_name = 's1_run'
    ppg_signal, fs = load_ppg_signal(record_name)

    print(f"Sinal carregado com {len(ppg_signal)} amostras e {fs} Hz")

    # Segmentar em janelas de 30 segundos
    segments = segment_signal(ppg_signal, fs, window_sec=30)
    
    print(f"Número de segmentos de 30s: {len(segments)}")

    # Salvar os segmentos
    for i, segment in enumerate(segments):
        save_path = os.path.join(OUTPUT_DIR, f"{record_name}_segment_{i:03d}.npy")
        np.save(save_path, segment)

    print(f"Segmentos salvos em: {OUTPUT_DIR}")
