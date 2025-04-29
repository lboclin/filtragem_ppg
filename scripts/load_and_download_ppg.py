import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt

# Configurações
SAVE_DIR = 'C:/Users/Usuário/Desktop/codes/projeto caes/filtragem_ppg/data/raw/'

def load_ppg_signal(record_name):
    record_path = os.path.join(SAVE_DIR, record_name)
    
    record = wfdb.rdrecord(record_path)

    print(f"Canais encontrados: {record.sig_name}")
    
    # Encontra o primeiro canal que tenha "PLETH"
    ppg_index = None
    for idx, signal_name in enumerate(record.sig_name):
        if 'PLETH' in signal_name.upper():
            ppg_index = idx
            break
    
    if ppg_index is None:
        raise ValueError(f"Canal PPG não encontrado no registro {record_name}")
    
    # Extrai o sinal PPG
    ppg_signal = record.p_signal[:, ppg_index]
    
    return ppg_signal, record.fs



if __name__ == "__main__":
    # Nome correto do arquivo
    example_record = 's1_run'
    ppg, fs = load_ppg_signal(example_record)

    print(f"Sinal PPG carregado! Duração: {len(ppg)/fs:.2f} segundos")

    # Plotar os primeiros 5 segundos do sinal
    t = np.arange(len(ppg)) / fs
    plt.plot(t[:5000], ppg[:5000])
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(f'PPG - {example_record}')
    plt.show()
