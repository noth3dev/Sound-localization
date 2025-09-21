
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import csv
import sys

# ---------- 설정 ----------
SAMPLE_RATE = 16000
WINDOW_SECONDS = 0.975
HOP_SECONDS = 0.5
THRESHOLD = 0.4
GAIN = 2.0  # 마이크 입력 증폭 배율

RISK_KEYWORDS = [
    "siren", "screaming", "scream", "gunshot", "gun shot", "gunfire",
    "alarm", "explosion", "glass", "breaking glass", "shout",
    "ambulance", "baby cry", "firework"
]

CLASS_MAP_PATH = "yamnet_class_map.csv"
# ---------------------------

q = queue.Queue()
stop_flag = threading.Event()

def load_class_map(path):
    class_names = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                class_names.append(row[2])
            else:
                class_names.append(row[-1])
    return class_names

def audio_callback(indata, frames, time_info, status):
    if status:
        print("녹음 상태:", status, file=sys.stderr)
    mono = indata[:, 0] if indata.ndim > 1 else indata
    mono = np.clip(mono * GAIN, -1.0, 1.0)  # 입력 증폭
    q.put(mono.copy())

def start_stream(device=None):
    blocksize = int(HOP_SECONDS * SAMPLE_RATE)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=blocksize,
        dtype='float32',
        channels=1,
        callback=audio_callback,
        device=device
    )
    stream.start()
    return stream

def main():
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    class_names = load_class_map(CLASS_MAP_PATH)
    print(f" {len(class_names)} gay")

    window_size = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop_size = int(HOP_SECONDS * SAMPLE_RATE)
    buffer = np.zeros(window_size, dtype='float32')

    try:
        stream = start_stream()
    except Exception as e:
        print("byeongsin:", e)
        return

    print("stream started")
    last_alert_time = 0.0
    ALERT_COOLDOWN = 1.0

    try:
        while True:
            chunk = q.get()
            if len(chunk) >= hop_size:
                buffer = np.concatenate((buffer[hop_size:], chunk[:hop_size]))
            else:
                buffer = np.concatenate((buffer[len(chunk):], chunk))

            waveform = buffer
            scores, embeddings, spectrogram = yamnet(waveform)
            avg_scores = tf.reduce_mean(scores, axis=0).numpy()

            top_idx = int(np.argmax(avg_scores))
            top_score = float(avg_scores[top_idx])
            top_name = class_names[top_idx] if 0 <= top_idx < len(class_names) else "Unknown"

            name_lower = top_name.lower()
            is_risk_by_name = any(k in name_lower for k in RISK_KEYWORDS)
            is_risk_by_score = top_score >= THRESHOLD

            now = time.time()
            if is_risk_by_name and is_risk_by_score:
                if now - last_alert_time > ALERT_COOLDOWN:
                    last_alert_time = now
                    print(f"[dan ger]{time.strftime('%H:%M:%S')} - {top_name} ({top_score:.2f})")
            else:
                print(f"{time.strftime('%H:%M:%S')} - {top_name} ({top_score:.2f})")

    except KeyboardInterrupt:
        print("\nc")
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
