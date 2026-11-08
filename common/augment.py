"""
Augmentation de données audio (amélioration face au manque de données : ~22 h d'entraînement).

- Perturbation de vitesse (0.9 / 1.0 / 1.1), comme dans les recettes Kaldi/ESPnet.
- Ajout de bruit à un SNR aléatoire (bruit blanc/rose, ou fichiers de bruit réels type MUSAN).
- Variation de gain.
Le SpecAugment (masquage temps/fréquence) est activé directement dans la config des modèles
(voir common/asr.py), il n'est donc pas ici.

Appliquée uniquement au split train, à la volée dans le collator : chaque epoch voit une version différente.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def speed_perturb(wav: np.ndarray, factor: float) -> np.ndarray:
    """Change vitesse ET hauteur (comme `sox speed`), par rééchantillonnage."""
    if abs(factor - 1.0) < 1e-6:
        return wav
    from scipy.signal import resample
    n = int(round(len(wav) / factor))
    return resample(wav, n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bruit rose (1/f), plus réaliste qu'un bruit blanc pour simuler une salle."""
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    freqs = np.arange(len(f))
    freqs[0] = 1
    pink = np.fft.irfft(f / np.sqrt(freqs), n)
    return (pink / (np.std(pink) + 1e-8)).astype(np.float32)


def add_noise(wav: np.ndarray, snr_db: float, noise: np.ndarray | None = None,
              rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if noise is None:
        noise = pink_noise(len(wav), rng)
    elif len(noise) < len(wav):
        noise = np.tile(noise, int(np.ceil(len(wav) / len(noise))))[: len(wav)]
    else:
        start = rng.integers(0, len(noise) - len(wav) + 1)
        noise = noise[start:start + len(wav)]
    p_sig = np.mean(wav ** 2) + 1e-10
    p_noise = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return (wav + scale * noise).astype(np.float32)


class Augmenter:
    """Augmentation aléatoire, appelée sur chaque exemple d'entraînement."""

    def __init__(self, p_speed: float = 0.5, speeds=(0.9, 1.1), p_noise: float = 0.4,
                 snr_range=(10.0, 30.0), p_gain: float = 0.3, gain_db=(-6.0, 6.0),
                 noise_dir: str | Path | None = None, sr: int = 16_000, seed: int = 42):
        self.p_speed, self.speeds = p_speed, speeds
        self.p_noise, self.snr_range = p_noise, snr_range
        self.p_gain, self.gain_db = p_gain, gain_db
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.noises = []
        if noise_dir:
            import soundfile as sf
            for f in list(Path(noise_dir).rglob("*.wav"))[:200]:
                w, s = sf.read(str(f), dtype="float32")
                if s == sr:
                    self.noises.append(w if w.ndim == 1 else w.mean(1))

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if self.py_rng.random() < self.p_speed:
            wav = speed_perturb(wav, self.py_rng.choice(self.speeds))
        if self.py_rng.random() < self.p_noise:
            noise = self.py_rng.choice(self.noises) if self.noises else None
            wav = add_noise(wav, self.py_rng.uniform(*self.snr_range), noise, self.rng)
        if self.py_rng.random() < self.p_gain:
            wav = wav * (10 ** (self.py_rng.uniform(*self.gain_db) / 20))
        peak = np.max(np.abs(wav)) + 1e-8
        return (wav / peak * 0.95).astype(np.float32) if peak > 1.0 else wav.astype(np.float32)
