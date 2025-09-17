#!/usr/bin/env python3
from __future__ import annotations
import os
import math
import numpy as np
from scipy.signal import get_window
from scipy.linalg import eigh
from scipy.io import wavfile

SAMPLE_RATE = 48000
BLOCK_SIZE = 4096
OVERLAP = 0.0
CHANNELS = 4
MIC_SPACING_M = 0.035
SPEED_OF_SOUND = 343.0
N_SOURCES_EST = 3
AZIMUTH_GRID_DEG = np.linspace(-90.0, 90.0, 361)

F_MIN = 500.0
F_MAX = 4000.0

COV_ALPHA = 0.9

SOURCE_FILES = ["1.mp3", "2.mp3", "3.mp3", "4.mp3"]
SOURCE_ANGLES_DEG = [-40.0, -10.0, 25.0, 55.0]

OUTPUT_DIR = "output"


def base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


try:
    import librosa
    def load_audio(path: str, sr: int) -> np.ndarray:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float64)
except Exception:
    try:
        from pydub import AudioSegment
        def load_audio(path: str, sr: int) -> np.ndarray:
            seg = AudioSegment.from_file(path)
            seg = seg.set_frame_rate(sr).set_channels(1)
            arr = np.array(seg.get_array_of_samples(), dtype=np.float32)
            peak = float(1 << (8 * seg.sample_width - 1))
            y = (arr / peak).astype(np.float64)
            return y
    except Exception:
        def load_audio(path: str, sr: int) -> np.ndarray:
            raise RuntimeError(
                "No MP3 loader available. Install one of: 'librosa' (pip install librosa) or 'pydub' + ffmpeg."
            )


def mic_positions_ula(num_mics: int, spacing_m: float) -> np.ndarray:
    idx = np.arange(num_mics) - (num_mics - 1) / 2.0
    return idx * spacing_m


def steering_vector_ula(theta_deg: float, freq_hz: float, mic_pos_x: np.ndarray) -> np.ndarray:
    theta_rad = np.deg2rad(theta_deg)
    tau = (mic_pos_x * np.sin(theta_rad)) / SPEED_OF_SOUND
    phase = -2.0 * np.pi * freq_hz * tau
    return np.exp(1j * phase).astype(np.complex128)


def freq_bins(fs: int, nfft: int, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], mask


def music_spectrum(Rxx_f_list: list[np.ndarray], freqs: np.ndarray, mic_pos_x: np.ndarray,
                   angle_grid_deg: np.ndarray, num_sources: int) -> np.ndarray:
    M = Rxx_f_list[0].shape[0]
    P = np.zeros_like(angle_grid_deg, dtype=np.float64)
    for fi, f in enumerate(freqs):
        Rxx = Rxx_f_list[fi]
        Rxx = 0.5 * (Rxx + Rxx.conj().T)
        w, V = eigh(Rxx)
        En = V[:, : max(M - num_sources, 1)]
        EnH = En.conj().T
        for ai, ang in enumerate(angle_grid_deg):
            a = steering_vector_ula(ang, f, mic_pos_x)
            denom = np.linalg.norm(EnH @ a)**2
            if denom <= 1e-12:
                denom = 1e-12
            P[ai] += 1.0 / denom
    P = P / np.max(P)
    return P


def find_top_k_angles(P: np.ndarray, angle_grid_deg: np.ndarray, k: int) -> list[float]:
    idx = np.argpartition(-P, k)[:k]
    idx = idx[np.argsort(-P[idx])]
    return [float(angle_grid_deg[i]) for i in idx]


def beamform_separate_block(block: np.ndarray, fs: int, angles_deg: list[float], mic_pos_x: np.ndarray,
                             window: np.ndarray | None = None) -> np.ndarray:
    N, M = block.shape
    nfft = N
    if window is None:
        window = np.ones(N, dtype=np.float64)
    win = window.reshape(-1, 1)

    X = np.fft.rfft(block * win, n=nfft, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)

    separated = np.zeros((len(angles_deg), N), dtype=np.float64)

    for k, ang in enumerate(angles_deg):
        Yf = np.zeros_like(X[:, 0], dtype=np.complex128)
        for fi, f in enumerate(freqs):
            a = steering_vector_ula(ang, f, mic_pos_x)
            Yf[fi] = np.sum(X[fi, :] * a.conj()) / M
        y = np.fft.irfft(Yf, n=nfft)
        separated[k, :] = np.real_if_close(y)
    return separated


def next_pow2(n: int) -> int:
    return 1 if n == 0 else 2**(int(n - 1).bit_length())


def simulate_array_from_sources(sources: list[np.ndarray], source_angles_deg: list[float],
                                mic_pos_x: np.ndarray, sr: int) -> np.ndarray:
    assert len(sources) == len(source_angles_deg)
    S = len(sources)
    M = mic_pos_x.shape[0]

    T = int(min(len(s) for s in sources))
    sources = [s[:T] for s in sources]

    delays_samples = np.zeros((M, S), dtype=np.float64)
    for si, ang in enumerate(source_angles_deg):
        theta = math.radians(ang)
        tau = (mic_pos_x * math.sin(theta)) / SPEED_OF_SOUND
        delays_samples[:, si] = tau * sr

    d_min = delays_samples.min()
    delays_shifted = delays_samples - d_min
    d_max = int(np.ceil(delays_shifted.max()))

    Nfft = next_pow2(T + d_max + 1)
    k = np.arange(Nfft//2 + 1, dtype=np.float64)

    Y_mics = np.zeros((M, Nfft), dtype=np.float64)

    for si, src in enumerate(sources):
        x = np.zeros(Nfft, dtype=np.float64)
        x[:T] = src
        X = np.fft.rfft(x, n=Nfft)
        for m in range(M):
            d = delays_shifted[m, si]
            phase = np.exp(-1j * 2.0 * np.pi * k * d / Nfft)
            Yf = X * phase
            y = np.fft.irfft(Yf, n=Nfft)
            Y_mics[m, :] += y

    Y_mics /= max(1, S)

    Y = Y_mics[:, :T].T
    return Y


def compute_covariance_and_music(mix: np.ndarray, fs: int, block_size: int,
                                 fmin: float, fmax: float, cov_alpha: float,
                                 angle_grid_deg: np.ndarray, num_est: int,
                                 mic_pos_x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    N, M = mix.shape
    window = get_window("hann", block_size, fftbins=True).astype(np.float64)
    freqs_all, mask = freq_bins(fs, block_size, fmin, fmax)
    F = len(freqs_all)
    Rxx_list = [1e-6 * np.eye(M, dtype=np.complex128) for _ in range(F)]

    hop = int(block_size * (1.0 - OVERLAP))
    for start in range(0, N - block_size + 1, hop):
        block = mix[start:start+block_size, :]
        X = np.fft.rfft(block * window.reshape(-1, 1), n=block_size, axis=0)
        X_band = X[mask, :]
        for fi in range(F):
            x = X_band[fi, :].reshape(M, 1)
            Rxx_inst = x @ x.conj().T
            Rxx_list[fi] = cov_alpha * Rxx_list[fi] + (1.0 - cov_alpha) * Rxx_inst

    P = music_spectrum(Rxx_list, freqs_all, mic_pos_x, angle_grid_deg, num_est)
    angles = find_top_k_angles(P, angle_grid_deg, num_est)
    return P, angles
def beamform_full(mix: np.ndarray, fs: int, angles_deg: list[float], mic_pos_x: np.ndarray,
                  block_size: int) -> np.ndarray:
    T, M = mix.shape
    window = get_window("hann", block_size, fftbins=True).astype(np.float64)
    hop = block_size
    K = len(angles_deg)
    out = np.zeros((K, T), dtype=np.float64)

    for start in range(0, T, hop):
        end = min(start + block_size, T)
        blk = np.zeros((block_size, M), dtype=np.float64)
        blk[:end-start, :] = mix[start:end, :]
        y_sep = beamform_separate_block(blk, fs, angles_deg, mic_pos_x, window)
        out[:, start:end] += y_sep[:, :end-start]
    return out


def main():
    bdir = base_dir()
    sound_dir = os.path.join(bdir, "sound")
    paths = [os.path.join(sound_dir, fn) for fn in SOURCE_FILES]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print("Missing files in ./sound:")
        for p in missing:
            print(" -", p)
        print("Please add 1.mp3, 2.mp3, 3.mp3, 4.mp3 to the 'sound' folder.")
        return

    print("Loading MP3s ...")
    sources = [load_audio(p, SAMPLE_RATE) for p in paths]

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x**2) + 1e-12))
    sources = [s / max(rms(s), 1e-6) for s in sources]

    M = CHANNELS
    mic_x = mic_positions_ula(M, MIC_SPACING_M)

    print("Simulating array mixture ({} mics) ...".format(M))
    mix = simulate_array_from_sources(sources, SOURCE_ANGLES_DEG, mic_x, SAMPLE_RATE)

    print("Estimating DoAs with MUSIC ...")
    P, angles = compute_covariance_and_music(
        mix, SAMPLE_RATE, BLOCK_SIZE, F_MIN, F_MAX, COV_ALPHA,
        AZIMUTH_GRID_DEG, N_SOURCES_EST, mic_x
    )
    print("Estimated angles (deg):", ", ".join(f"{a:.1f}" for a in angles))

    print("Beamforming separation ...")
    separated = beamform_full(mix, SAMPLE_RATE, angles, mic_x, BLOCK_SIZE)

    os.makedirs(os.path.join(bdir, OUTPUT_DIR), exist_ok=True)
    for i in range(separated.shape[0]):
        y = separated[i, :]
        y = y / max(np.max(np.abs(y)), 1e-6)
        wav_path = os.path.join(bdir, OUTPUT_DIR, f"separated_{i+1}.wav")
        wavfile.write(wav_path, SAMPLE_RATE, (y * 32767.0).astype(np.int16))
        print("Saved:", wav_path)

    print("Done.")


if __name__ == "__main__":
    main()
