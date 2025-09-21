
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import csv
import time
from scipy.signal import resample

SAMPLE_RATE = 16000
THRESHOLD = 0.4
RISK_KEYWORDS = [
    "siren", "screaming", "scream", "gunshot", "gun shot", "gunfire",
    "alarm", "explosion", "glass", "breaking glass", "shout",
    "ambulance", "baby cry", "firework"
]
CLASS_MAP_PATH = "yamnet_class_map.csv"
SOUND_FOLDER = "./sound"

def load_class_map(path):
    class_names = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            class_names.append(row[2] if len(row) >= 3 else row[-1])
    return class_names

def load_audio_wav(filename, target_sr=SAMPLE_RATE):
    wav, sr = sf.read(filename, dtype='float32')
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)  # 모노 변환
    if sr != target_sr:
        num_samples = int(len(wav) * target_sr / sr)
        wav = resample(wav, num_samples)
    return wav

def analyze_file(filename, yamnet, class_names):
    waveform = load_audio_wav(filename, SAMPLE_RATE)
    scores, embeddings, spectrogram = yamnet(waveform)
    avg_scores = tf.reduce_mean(scores, axis=0).numpy()

    top_idx = int(np.argmax(avg_scores))
    top_score = float(avg_scores[top_idx])
    top_name = class_names[top_idx] if 0 <= top_idx < len(class_names) else "Unknown"

    rms = np.sqrt(np.mean(waveform**2))
    name_lower = top_name.lower()
    is_risk_by_name = any(k in name_lower for k in RISK_KEYWORDS)
    is_risk_by_score = top_score >= THRESHOLD

    print(f"'{filename}'")
    print(f"RMS: {rms:.5f}")
    print(f"/{top_name} ({top_score:.2f})")
    if is_risk_by_name and is_risk_by_score:
        print("⚠️ WARNING: High probability of dangerous sound detected.")
    else:
        print("No significant dangerous sounds detected.")

def main():
    class_names = load_class_map(CLASS_MAP_PATH)
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    files = [f for f in os.listdir(SOUND_FOLDER) if f.lower().endswith(".wav")]
    if not files:
        print(f"'{SOUND_FOLDER}' file nono")
        return

    print(f"{len(files)} gay\n")
    for f in files:
        analyze_file(os.path.join(SOUND_FOLDER, f), yamnet, class_names)

if __name__ == "__main__":
    main()
