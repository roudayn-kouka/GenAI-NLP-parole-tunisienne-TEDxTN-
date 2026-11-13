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
    assert text.normalize_text("الfumoir") == "الfumoir"                      # évaluation : pas de coupure
    assert text.normalize_text("الfumoir", text.WRITING_NORM) == "ال fumoir"  # convention d'écriture (notebook 03)
    n2, trace = text.normalize_text(s, return_trace=True)
    assert n2 == n and len(trace) >= 3


def test_hesitations():
    """Hésitations repérées sur la forme d'origine : « أم » est un mot (gardé), « آم » une hésitation (retirée),
    même si la normalisation de l'alef les rend identiques ensuite (ام)."""
    nohes = text.EVAL_NORM_NOHES
    assert text.normalize_text("أم الدنيا", nohes) == "ام الدنيا"
    assert text.normalize_text("آم الدنيا", nohes) == "الدنيا"
    assert text.normalize_text("آم الدنيا") == "ام الدنيا"                   # EVAL_NORM : hésitations gardées
    assert text.normalize_text("آ، إم أه اه أ إ euh ممممم باهي", nohes) == "باهي"
    assert text.normalize_text("أنا أول واحد", nohes) == "انا اول واحد"      # mots commençant par أ gardés
    assert not text.EVAL_NORM.remove_hesitations and not text.EVAL_NORM.split_mixed_tokens


def test_pipeline_names():
    """Nom des cascades avec ASR source ; refus d'une cascade faite avec un autre ASR que l'asr_best."""
    from common import data
    n = data.pipeline_name("C2", "whisper-small-ft", "nllb-lora")
    assert n == "C2-[whisper-small-ft]->nllb-lora" and data.pipeline_source(n) == "whisper-small-ft"
    assert ">" not in data.pred_path("ast", n, "test").name                  # nom de fichier valide sous Windows
    assert data.stale_pipeline(n, asr_best="whisper-small-ft") is None
    assert data.stale_pipeline(n, asr_best="w2v-bert-2.0-ft+kenlm")           # autre ASR : refusé
    assert data.stale_pipeline("C4-[whisper-small-ft+llm]->llm", asr_best="whisper-small-ft") is None
    assert data.stale_pipeline("C2-[ref]->nllb-lora", asr_best="autre") is None   # oracle : indépendant de l'ASR
    assert data.stale_pipeline("C2-asr->nllb-ft", asr_best="whisper-small-ft")   # ancien format : refusé
    assert data.stale_pipeline("E2E-2-whisper-small-from-asr", asr_best="x") is None


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
    # règle de la post-correction : retenue seulement si l'IC apparié de la différence de WER est sous 0
    R = ["a b c d", "e f g h", "i j k l", "m n o p"] * 5
    bad = ["a x c d", "e f y h", "i j k z", "w n o p"] * 5
    assert metrics.correction_helps(R, bad, R, n_samples=200)[0]        # la correction répare tout : retenue
    assert not metrics.correction_helps(R, R, bad, n_samples=200)[0]    # elle dégrade : refusée
    assert not metrics.correction_helps(R, bad, bad, n_samples=200)[0]  # aucun effet (IC = [0 ; 0]) : refusée


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


class FakeLLM:
    name = "fake-llm"

    def correct(self, texts):
        return [t + " باهي" for t in texts], [True for _ in texts]

    def translate(self, texts, target_name):
        return [f"[{target_name}] llm" for _ in texts]


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
    # options LLM : l'avertissement (mesures du dev) apparaît dans l'étape concernée, marquée « warn »
    from common.pipeline import LLM_CORRECTION_WARNING, LLM_TRANSLATION_WARNING
    import html as _html
    pipe = TunisianSpeechPipeline(asr=FakeASR(), e2e=FakeE2E(), mt=FakeMT(), llm=FakeLLM())
    res = pipe.run((44_100, (np.repeat(wav[:, None], 2, 1) * 32767).astype(np.int16)), target="Anglais", mode="pipeline", use_llm=True,
                   mt_engine="NLLB + LLM")
    st = {s.key: s for s in res.steps}
    assert st["llm"].status == "warn" and _html.escape(LLM_CORRECTION_WARNING) in st["llm"].html
    assert st["translate"].status == "warn" and _html.escape(LLM_TRANSLATION_WARNING) in st["translate"].html
    assert set(res.translations) == {"Pipeline ASR → NLLB", "Pipeline ASR → LLM"}


def test_history():
    """Historique de l'interface : enregistrer, lister, rouvrir (parcours complet sans modèle), exporter, supprimer."""
    import json
    import tempfile
    from app import history
    pipe = TunisianSpeechPipeline(asr=FakeASR(), e2e=FakeE2E(), mt=FakeMT())
    res = pipe.run((44_100, (np.repeat(_tone(3, 0.3)[:, None], 2, 1) * 32767).astype(np.int16)), target="Anglais",
                   mode="compare")
    with tempfile.TemporaryDirectory() as d:
        rid = history.save(res, {"source": "micro", "source_kind": "micro", "settings": {"mode": "Comparer"}}, d)
        recs = history.list_records(d)
        assert [r["id"] for r in recs] == [rid] and recs[0]["source"] == "micro" and recs[0]["mode"] == "Comparer"
        back = history.to_result(history.load(rid, d))
        assert back.transcript == res.transcript and back.translations == res.translations
        assert [s.key for s in back.steps] == [s.key for s in res.steps]
        assert [s.model for s in back.steps] == [s.model for s in res.steps]
        assert any("data:image/png;base64," in s.html for s in back.steps)   # figures conservées (PNG)
        assert json.load(open(history.export_path(rid, d), encoding="utf-8"))["id"] == rid
        rid2 = history.save(res, {"source": "corpus test__00001"}, d)
        assert len(history.list_records(d)) == 2
        assert history.delete(rid, d) and not history.delete(rid, d)
        assert [r["id"] for r in history.list_records(d)] == [rid2]
        for bad in ("../x", "abc", "20261113-120000-zzzzzz"):
            try:
                history.load(bad, d)
                raise AssertionError("identifiant invalide accepté")
            except ValueError:
                pass


def test_reference_scores():
    """Onglet « Voix existantes » : scores d'un segment contre sa référence (mêmes conventions que l'évaluation)."""
    from app.scoring import compare_to_reference, reference_html
    ref_tun, ref_en = "آنا نحب tunis برشا", "I love Tunis a lot"
    s = compare_to_reference(ref_tun, {"Directe": ref_en}, ref_tun, ref_en)
    assert s["WER"] == 0 and s["CER"] == 0
    assert s["translations"]["Directe"]["BLEU"] == 100 and s["translations"]["Directe"]["chrF++"] == 100
    s = compare_to_reference("آنا نحب برشا", {"Pipeline": "I like it"}, ref_tun, ref_en)
    assert s["WER"] == 25.0   # une suppression sur 4 mots
    assert 0 <= s["translations"]["Pipeline"]["BLEU"] < 100 and s["translations"]["Pipeline"]["chrF++"] < 100
    # même WER que l'évaluation des notebooks (RAW_NORM : ponctuation et casse latine ignorées)
    assert compare_to_reference("Tunis, برشا!", {}, "tunis برشا", "")["WER"] == 0
    h = reference_html(s, "آنا نحب برشا", {"Pipeline": "I like it"}, ref_tun, ref_en)
    assert "<del>" in h and "WER" in h and "BLEU" in h
    # tableau compact des segments : filtres, colonnes, identifiant en première colonne (sert à la sélection)
    import pandas as pd
    from app.app import VOICE_COLS, filter_voices, voice_table
    v = pd.DataFrame({"seg_id": ["test__1", "test__2", "dev__1"], "split": ["test", "test", "dev"],
                      "talk": ["a", "b", "a"], "seg_type": ["TUN", "FOR", "TUN"], "duration": [3.0, 12.0, 5.0],
                      "transcription": ["آنا " * 30, "bonjour", "x"], "translation": ["", "", ""], "path": ["", "", ""]})
    t = voice_table(filter_voices(v, "test", "Tous", ["TUN"], 0, 30))
    assert list(t.columns) == VOICE_COLS and t["n°"].tolist() == ["1"] and t.iloc[0, -1].endswith("…")
    assert voice_table(filter_voices(v, "test", "a", ["FOR"], 0, 30)).empty


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
