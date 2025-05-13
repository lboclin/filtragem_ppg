import numpy as np
import os

def add_baseline_wander(signal, fs, amplitude=0.1, freq=0.3):
    t = np.arange(len(signal)) / fs
    drift = amplitude * np.sin(2 * np.pi * freq * t)
    return signal + drift

def add_motion_artifact(signal, fs, burst_duration=0.5, burst_amplitude=0.5):
    samples = len(signal)
    artifact = np.zeros(samples)
    burst_len = int(burst_duration * fs)
    for _ in range(np.random.randint(2, 5)):
        start = np.random.randint(0, samples - burst_len)
        artifact[start:start+burst_len] += burst_amplitude * (np.random.rand(burst_len) - 0.5)
    return signal + artifact

def add_sensor_dropout(signal, fs, drop_duration=0.3):
    signal = signal.copy()
    drop_len = int(drop_duration * fs)
    for _ in range(np.random.randint(1, 3)):
        start = np.random.randint(0, len(signal) - drop_len)
        signal[start:start+drop_len] = 0
    return signal

INPUT_DIR = r"C:\Users\Usuário\Desktop\codes\projeto caes\filtragem_ppg\data\processed"
OUTPUT_DIR = r"C:\Users\Usuário\Desktop\codes\projeto caes\filtragem_ppg\data\noisy_realistic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

fs = 500  # Hz

for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".npy"):
        signal = np.load(os.path.join(INPUT_DIR, fname))
        noisy = add_baseline_wander(signal, fs)
        noisy = add_motion_artifact(noisy, fs)
        noisy = add_sensor_dropout(noisy, fs)
        out_path = os.path.join(OUTPUT_DIR, fname.replace(".npy", "_realistic.npy"))
        np.save(out_path, noisy)
        print(f"✅ {fname} salvo com ruído realista.")
