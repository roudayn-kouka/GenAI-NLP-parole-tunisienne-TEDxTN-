"""Tests du code partagé sans GPU ni téléchargement : données synthétiques au format réel de TEDxTN.
Lancer :  python -m pytest -q tests/   (ou  python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import metrics, text  # noqa: E402
from common.pipeline import TunisianSpeechPipeline, energy_vad, log_mel  # noqa: E402

SR = 16_000


def test_script_and_cmi():
    s = "أول حاجة كان تعرفوه base العباد الكل عندها cube قبل الfumoir"
    tags = dict(text.tag_tokens(s))
    assert tags["base"] == "lat" and tags["الfumoir"] == "mixed" and tags["حاجة"] == "ar"
    assert text.segment_type(s) == "MIXED"
    assert text.segment_type("Mais bon it happens") == "FOR"
    assert text.segment_type("باهي") == "TUN"
    assert 0 < text.cmi(s) < 50
    assert text.cmi("باهي برشا") == 0


def test_normalization():
    s = "أنا مشيت للـ hôtel برشاااا !"
    n = text.normalize_text(s)
    assert "أ" not in n and "!" not in n and "برشا" in n and "ـ" not in n
    assert text.normalize_text("الfumoir") == "ال fumoir"
    n2, trace = text.normalize_text(s, return_trace=True)
    assert n2 == n and len(trace) >= 3


def test_cross_script():
    assert text.is_script_variant("circulation", "سيركيلاسيون")
    assert text.is_script_variant("الفيموار", "fumoir")
    assert not text.is_script_variant("كتاب", "circulation")


def test_metrics():
    refs = ["آنا محمود شنعمل بالضبط", "مشيت ل hôtel", "باهي"]
    hyps = ["انا محمود شنعمل بالظبط", "مشيت ل هوتال", "باهي"]
    raw = metrics.wer_cer(refs, hyps, text.RAW_NORM)
    norm = metrics.wer_cer(refs, hyps, text.EVAL_NORM)
    assert norm["WER"] < raw["WER"]  # la normalisation pardonne la variante d'alef
    errs = metrics.error_table(refs, hyps)
    assert set(errs.op) <= {"S", "D", "I"} and len(errs) >= 2
    cats = set(errs.categorie)
    assert "même mot, autre script" in cats or "mot emprunté / code-switching" in cats
    by = metrics.wer_by_segment_type(refs, hyps)
    assert "ALL" in set(by.type)
    sc = metrics.translation_scores(["hello everyone", "let me get straight"], ["Hello everyone", "so let me get straight to the topic"])
    assert sc["BLEU"] > 0 and sc["chrF++"] > 0
    m, lo, hi = metrics.bootstrap_ci(metrics.wer_fn(), hyps, refs, n_samples=50)
    assert lo <= m <= hi
    assert metrics.translation_error_flags("x", "hello hello hello hello hello", "good morning everyone")


def _tone(seconds, gap=0.0):
    t = np.arange(int(seconds * SR)) / SR
    sig = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    return np.concatenate([np.zeros(int(gap * SR)), sig]).astype(np.float32)


def test_vad_and_features():
    wav = np.concatenate([_tone(2, 0.5), _tone(1.5, 1.0), np.zeros(SR // 2, np.float32)])
    regions, chunks, e, thr = energy_vad(wav)
    assert len(regions) == 2 and len(chunks) >= 1
    long = np.concatenate([_tone(20, 0.5), _tone(20, 1.0)])
    _, chunks2, _, _ = energy_vad(long)
    assert all(b - a <= 25.01 for a, b in chunks2) and len(chunks2) >= 2
    mel = log_mel(wav)
    assert mel.shape[0] == 80 and abs(mel.shape[1] - len(wav) / 160) < 5


class FakeASR:
    name, kind, warning = "fake-whisper", "whisper", ""

    def describe(self):
        return {"modèle": self.name}

    def transcribe(self, wavs):
        return [{"text": "أول حاجة الfumoir base برشاااا", "tokens": ["أول", " حاجة", " الfumoir"],
                 "token_probs": [0.9, 0.6, 0.3], "confidence": 0.6} for _ in wavs]


class FakeE2E:
    name, warning = "fake-ast", ""

    def translate_audio(self, wavs):
        return [{"text": "First thing, the smoking room"} for _ in wavs]


class FakeMT:
    name, warning = "fake-nllb", ""

    def translate(self, texts, tgt, src_code=None):
        return [f"[{tgt}] first thing the smoking room" for _ in texts]


def test_pipeline_with_fake_backends():
    pipe = TunisianSpeechPipeline(asr=FakeASR(), e2e=FakeE2E(), mt=FakeMT())
    wav = _tone(3, 0.3)
    for mode in ("pipeline", "direct", "compare"):
        for target in ("Anglais", "Français", "Arabe standard (MSA)"):
            res = pipe.run((44_100, (np.repeat(wav[:, None], 2, 1) * 32767).astype(np.int16)), target=target, mode=mode)
            keys = [s.key for s in res.steps]
            assert keys[:3] == ["input", "vad", "features"] and keys[-1] == "result", keys
            assert all(s.status != "error" for s in res.steps), [(s.key, s.summary) for s in res.steps]
            assert res.transcript and res.translations
            if mode == "compare":
                assert len(res.translations) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
