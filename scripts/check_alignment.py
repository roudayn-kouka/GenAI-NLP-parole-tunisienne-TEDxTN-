"""Contrôle d'alignement audio / annotations, après téléchargement (scripts/prepare_data.py).

    python scripts/check_alignment.py                 # contrôle + correction des décalages + redécoupage
    python scripts/check_alignment.py --dry-run       # contrôle seul, youtube_urls.csv inchangé

Principe : pour chaque talk, 3 segments (début, milieu, fin) sont transcrits par Whisper zero-shot et comparés à la
référence (CER normalisé). Le zero-shot est mauvais sur le tunisien, mais un segment bien aligné reste nettement
meilleur qu'un segment décalé ou qu'une mauvaise vidéo. Pour un talk au CER anormal, on essaie des décalages de
-10 s à +10 s par pas de 0,5 s. Décision :
- décalage constant (le même pour les 3 segments, à ±1 s) qui fait gagner au moins 15 points de CER : écrit dans
  youtube_urls.csv (colonne offset_s, appliquée par data.cut_segments) ;
- CER encore ≥ 95 % au meilleur décalage (transcription sans rapport avec la référence) : « exclu » ;
- sinon : gardé (CER élevé dû à l'orateur ou au code-switching, aucun décalage démontré).

Sorties : results/alignment_check.csv (par segment et par décalage testé), results/alignment_summary.csv (par talk),
workdir/data/audio_report.csv et corpus.parquet mis à jour, heures utilisables par split.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data, metrics, text  # noqa: E402
from common.config import AUDIO_DIR, BASE_MODELS, DATA_DIR, PAPER_SPLIT_STATS, RESULTS_DIR, SAMPLING_RATE  # noqa: E402

PAPER_HOURS = {"train": 22 + 40 / 60, "dev": 1 + 7 / 60, "test": 1 + 14 / 60}   # Table 5 du papier


def pick_segments(g: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Début, milieu, fin du talk, parmi les segments assez longs pour un CER stable."""
    c = g[(g.duration.between(3, 20)) & (g.n_chars >= 15)].sort_values("start_time")
    if len(c) < n:
        c = g.sort_values("start_time")
    idx = np.unique(np.linspace(0, len(c) - 1, n).round().astype(int))
    return c.iloc[idx]


def seg_cer(ref: str, hyp: str) -> float:
    """CER normalisé d'un segment, plafonné à 100 (les hallucinations de Whisper dépassent souvent 100 %)."""
    v = metrics.wer_cer([ref], [hyp], text.EVAL_NORM)["CER"]
    return 100.0 if not np.isfinite(v) else min(100.0, v)


class Transcriber:
    def __init__(self, model_id: str, batch_size: int):
        import torch
        from common.asr import load_whisper
        self.proc, self.model = load_whisper(model_id, for_training=False)
        if torch.cuda.is_available():
            self.model = self.model.half()
        self.bs = batch_size

    def __call__(self, wavs):
        from common.asr import whisper_generate
        return whisper_generate(self.model, self.proc, wavs, batch_size=self.bs, max_new_tokens=128)


def clips(wav: np.ndarray, segs: pd.DataFrame, offset: float) -> list[np.ndarray]:
    out = []
    for s in segs.itertuples():
        a = max(0, int((s.start_time + offset) * SAMPLING_RATE))
        b = min(len(wav), int((s.end_time + offset) * SAMPLING_RATE))
        out.append(wav[a:b] if b - a > SAMPLING_RATE // 10 else np.zeros(SAMPLING_RATE // 10, np.float32))
    return out


def energy_check(df: pd.DataFrame, n: int = 5, seed: int = 0) -> pd.DataFrame:
    """Contrôle qualité sans écoute : durée du fichier découpé = durée annotée, énergie non nulle."""
    import soundfile as sf
    rows = []
    for s in df.sample(n, random_state=seed).itertuples():
        w, sr = sf.read(s.path, dtype="float32")
        rms = float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0
        rows.append({"seg_id": s.seg_id, "split": s.split, "duree_annotee_s": s.duration,
                     "duree_fichier_s": round(len(w) / sr, 3), "sr": sr, "canaux": 1 if w.ndim == 1 else w.shape[1],
                     "rms_dBFS": round(20 * np.log10(rms + 1e-12), 1),
                     "part_silence_%": round(100 * float(np.mean(np.abs(w) < 1e-3)), 1),
                     "n_mots_ref": s.n_words})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODELS["whisper-large-v3-turbo"])
    ap.add_argument("--n-seg", type=int, default=3)
    ap.add_argument("--max-offset", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--margin", type=float, default=20.0,
                    help="un talk est anormal si son CER dépasse la médiane des talks de plus de MARGIN points")
    ap.add_argument("--min-gain", type=float, default=15.0,
                    help="gain minimal de CER (points) pour accepter un décalage")
    ap.add_argument("--exclude-cer", type=float, default=95.0,
                    help="CER (plafonné à 100) au meilleur décalage au-delà duquel le talk est exclu")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import soundfile as sf
    df = data.load_corpus(download=False)
    urls = pd.read_csv(data.URL_MAP_FILE, keep_default_na=False)
    offsets = data.talk_offsets()
    talks = [t for t in sorted(df.talk.unique()) if (AUDIO_DIR / f"{t}.wav").exists()]
    print(f"{len(talks)} talks avec audio / {df.talk.nunique()}")
    tr = Transcriber(a.model, a.batch_size)

    # 1) CER au décalage actuel
    rows, cache = [], {}
    for t in talks:
        g = df[df.talk == t]
        segs = pick_segments(g, a.n_seg)
        wav, sr = sf.read(str(AUDIO_DIR / f"{t}.wav"), dtype="float32")
        assert sr == SAMPLING_RATE, (t, sr)
        cache[t] = (wav, segs)
        off = float(offsets.get(t, 0.0))
        for s, h in zip(segs.itertuples(), tr(clips(wav, segs, off))):
            rows.append({"talk": t, "split": s.split, "seg_id": s.seg_id, "start_time": s.start_time,
                         "offset_s": off, "passe": "initiale", "ref": s.transcription, "hyp": h,
                         "CER": seg_cer(s.transcription, h)})
    res = pd.DataFrame(rows)
    per_talk = res.groupby("talk").CER.mean()
    med = float(per_talk.median())
    thr = med + a.margin
    print(f"CER zero-shot par talk : médiane {med:.1f} %, seuil d'anomalie {thr:.1f} %")
    suspects = per_talk[per_talk > thr].index.tolist()
    print(f"{len(suspects)} talks au CER anormal : {suspects}")

    # 2) Recherche d'un décalage constant pour les talks anormaux
    grid = np.round(np.arange(-a.max_offset, a.max_offset + 1e-9, a.step), 2)
    summary = []
    for t in talks:
        cur = float(offsets.get(t, 0.0))
        c0 = float(per_talk[t])
        decision = {"talk": t, "split": df[df.talk == t].split.iloc[0], "CER_initial": round(c0, 1),
                    "offset_initial": cur, "offset_final": cur, "CER_final": round(c0, 1), "decision": "OK"}
        if t in suspects:
            wav, segs = cache[t]
            grows = []
            for d in grid:
                hyps = tr(clips(wav, segs, cur + d))
                for s, h in zip(segs.itertuples(), hyps):
                    grows.append({"talk": t, "split": s.split, "seg_id": s.seg_id, "start_time": s.start_time,
                                  "offset_s": cur + d, "passe": "grille", "ref": s.transcription, "hyp": h,
                                  "CER": seg_cer(s.transcription, h)})
            gr = pd.DataFrame(grows)
            res = pd.concat([res, gr], ignore_index=True)
            mean_by_off = gr.groupby("offset_s").CER.mean()
            best_off, best_cer = float(mean_by_off.idxmin()), float(mean_by_off.min())
            per_seg_best = gr.loc[gr.groupby("seg_id").CER.idxmin(), "offset_s"].values
            constant = bool(np.all(np.abs(per_seg_best - best_off) <= 1.0))
            decision.update({"offset_meilleur": best_off, "CER_meilleur": round(best_cer, 1),
                             "offsets_par_segment": " ".join(f"{o:+.1f}" for o in per_seg_best),
                             "decalage_constant": constant})
            gain = c0 - best_cer
            # 1) décalage constant : même meilleur décalage pour les 3 segments et gain net de CER
            if constant and abs(best_off - cur) >= a.step and gain >= a.min_gain:
                decision.update({"offset_final": best_off, "CER_final": round(best_cer, 1),
                                 "decision": f"décalage corrigé ({best_off - cur:+.1f} s)"})
            # 2) irrécupérable : même au meilleur décalage, la transcription n'a rien à voir avec la référence
            elif best_cer >= a.exclude_cer:
                decision["decision"] = "exclu (irrécupérable)"
            # 3) sinon le talk est gardé : CER élevé (orateur, débit, code-switching) mais pas de décalage démontré
            else:
                decision["decision"] = "OK (CER élevé, pas de décalage démontré)"
        summary.append(decision)
        print(f"  {decision['decision']:45s} {t}  CER {decision['CER_initial']} -> {decision['CER_final']}")
    summ = pd.DataFrame(summary)
    res.to_csv(RESULTS_DIR / "alignment_check.csv", index=False)
    summ.to_csv(RESULTS_DIR / "alignment_summary.csv", index=False)
    print(f"-> {RESULTS_DIR / 'alignment_summary.csv'}")
    if a.dry_run:
        return

    # 3) Application : offset_s et exclusions dans youtube_urls.csv, puis redécoupage
    for d in summary:
        m = urls.audio_filename == d["talk"]
        urls.loc[m, "offset_s"] = d["offset_final"]
        if d["decision"].startswith("exclu"):
            urls.loc[m, "statut"] = "exclu"
            urls.loc[m, "motifs"] = (f"alignement : CER zero-shot {d['CER_initial']} % "
                                     f"(meilleur {d.get('CER_meilleur')} % à {d.get('offset_meilleur')} s)")
    urls.to_csv(data.URL_MAP_FILE, index=False)
    cut = data.cut_segments(df)
    cut.to_parquet(DATA_DIR / "corpus.parquet", index=False)

    rep_path = DATA_DIR / "audio_report.csv"
    rep = pd.read_csv(rep_path) if rep_path.exists() else pd.DataFrame({"talk": talks})
    rep = rep.drop(columns=[c for c in ("offset_s", "CER_initial", "CER_final", "alignement") if c in rep]).merge(
        summ[["talk", "offset_final", "CER_initial", "CER_final", "decision"]].rename(
            columns={"offset_final": "offset_s", "decision": "alignement"}), on="talk", how="left")
    rep.to_csv(rep_path, index=False)

    hours = []
    for split in ("train", "dev", "test"):
        d_all, d_ok = df[df.split == split], cut[cut.split == split]
        hours.append({"split": split, "talks_annotés": d_all.talk.nunique(), "talks_utilisables": d_ok.talk.nunique(),
                      "segments": len(d_ok), "segments_annotés": len(d_all),
                      "heures_utilisables": round(d_ok.duration.sum() / 3600, 2),
                      "heures_annotées": round(d_all.duration.sum() / 3600, 2),
                      "heures_papier": round(PAPER_HOURS[split], 2)})
    hours = pd.DataFrame(hours)
    hours.to_csv(RESULTS_DIR / "data_hours.csv", index=False)
    print(hours.to_string(index=False))
    excl = summ[summ.decision.str.startswith("exclu")].talk.tolist()
    missing = sorted(set(df.talk) - set(cut.talk))
    print(f"Talks exclus par l'alignement : {excl}")
    print(f"Talks sans audio utilisable : {missing}")
    print("Contrôle énergie / durée sur 5 segments aléatoires :")
    print(energy_check(cut).to_string(index=False))


if __name__ == "__main__":
    main()
