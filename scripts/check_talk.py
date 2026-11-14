"""Contrôle d'un talk au WER anormal (aucun entraînement, aucun système modifié, splits inchangés).

    python scripts/check_talk.py --talk Youssef_Cherif_Archelogie_TEDxENSI

Trois hypothèses sont testées pour expliquer un WER aberrant :
1. mauvaise vidéo : titre de la vidéo, durée de l'audio contre la fin de la dernière annotation ;
2. décalage : CER zero-shot (whisper-large-v3-turbo) des segments de l'échantillon, puis recherche d'un décalage
   de -MAX à +MAX s par segment ; un décalage réel donne le même meilleur décalage pour tous les segments ;
3. hallucinations : le modèle fine-tuné (asr_best) retranscrit le même échantillon que le notebook 01 (40 segments
   tirés avec SEED dans train_clean.parquet) ; on mesure la longueur de la sortie par rapport à la référence et
   les répétitions (part des trigrammes répétés), et si la sortie atteint la limite de tokens.
Plus un contrôle sans écoute de 3 segments (les 3 pires WER) : durée du fichier contre durée annotée, énergie,
part de silence, part de trames de parole.

Sorties : results/talk_check_<talk>.csv (par segment) et results/talk_check_<talk>_offsets.csv (grille).
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from common import data, metrics  # noqa: E402
from common.config import AUDIO_DIR, BASE_MODELS, DATA_DIR, MODELS_DIR, RESULTS_DIR, SAMPLING_RATE, SEED  # noqa: E402
from check_alignment import Transcriber, clips, seg_cer  # noqa: E402

MAX_NEW_TOKENS = 225   # limite de whisper_generate (notebook 01)


def repetition(t: str) -> float:
    """Part des trigrammes de mots qui sont des répétitions (0 = aucun, proche de 1 = boucle)."""
    w = t.split()
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    if not tri:
        return 0.0
    c = Counter(tri)
    return float(sum(n - 1 for n in c.values()) / len(tri))


def energy(path: str, win: float = 0.025) -> dict:
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32")
    rms = float(np.sqrt(np.mean(w ** 2))) if len(w) else 0.0
    n = int(win * sr)
    e = 10 * np.log10((w[: len(w) // n * n].reshape(-1, n) ** 2).mean(1) + 1e-10) if len(w) >= n else np.array([-100.0])
    # trame de parole : à moins de 25 dB de la trame la plus forte du segment
    return {"duree_fichier_s": round(len(w) / sr, 3), "sr": sr, "rms_dBFS": round(20 * np.log10(rms + 1e-12), 1),
            "part_silence_%": round(100 * float(np.mean(np.abs(w) < 1e-3)), 1),
            "part_trames_parole_%": round(100 * float(np.mean(e > e.max() - 25)), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talk", required=True)
    ap.add_argument("--ft-model", default=str(MODELS_DIR / "asr_whisper-small" / "final"))
    ap.add_argument("--zs-model", default=BASE_MODELS["whisper-large-v3-turbo"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-offset", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--wide", type=float, default=0.0, help="recherche large ±WIDE s avec le modèle fine-tuné (0 = non)")
    ap.add_argument("--wide-step", type=float, default=2.0)
    ap.add_argument("--wide-n", type=int, default=20)
    ap.add_argument("--wide-only", action="store_true", help="seulement la recherche large (étapes 1 à 4 déjà faites)")
    ap.add_argument("--confirm", type=float, default=None,
                    help="contrôle indépendant d'un décalage (s) : CER zero-shot de l'échantillon à 0 et à ce décalage")
    a = ap.parse_args()
    if a.confirm is not None:
        confirm(a)
        return
    if a.wide_only:
        import soundfile as sf
        T = a.talk
        g_all = data.load_corpus(download=False).query("talk == @T")
        full, _ = sf.read(str(AUDIO_DIR / f"{T}.wav"), dtype="float32")
        wide_search(a, T, "_".join(T.split("_")[:2]), g_all, full, float(data.talk_offsets().get(T, 0.0)))
        return
    import soundfile as sf
    import torch
    from common import asr as A

    T = a.talk
    short = "_".join(T.split("_")[:2])
    # 1) vidéo
    urls = pd.read_csv(data.URL_MAP_FILE, keep_default_na=False)
    u = urls[urls.audio_filename == T].iloc[0].to_dict()
    corpus = data.load_corpus(download=False)
    g_all = corpus[corpus.talk == T]
    wav_path = AUDIO_DIR / f"{T}.wav"
    audio_s = sf.info(str(wav_path)).duration
    print("== 1. Vidéo")
    for k, v in u.items():
        print(f"   {k} : {v}")
    print(f"   audio : {audio_s:.1f} s ; annotations : {len(g_all)} segments, de {g_all.start_time.min():.1f} s "
          f"à {g_all.end_time.max():.1f} s ({g_all.duration.sum() / 60:.1f} min annotées)")

    # 2) même échantillon que le notebook 01 (cellule 51) : 40 segments du talk dans train_clean, random_state=SEED
    tc = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    s = tc[tc.talk == T]
    s = s.sample(min(a.n, len(s)), random_state=SEED).reset_index(drop=True)
    wavs = [data.load_audio(p) for p in s.path]

    # 3) modèle fine-tuné : hallucinations ?
    proc, model = A.load_whisper(a.ft_model, for_training=False)
    hyp = A.whisper_generate(model, proc, wavs, batch_size=a.batch_size)
    ntok = [len(proc.tokenizer(h, add_special_tokens=False).input_ids) for h in hyp]
    del model
    torch.cuda.empty_cache()
    m = metrics.wer_cer(s.transcription.tolist(), hyp)
    print(f"== 2. Modèle fine-tuné ({a.ft_model}) sur {len(s)} segments : WER {m['WER']:.2f} %, CER {m['CER']:.2f} % "
          f"(01_hard_talks.csv : 267.14 / 137.29)")
    rows = []
    for r, h, k in zip(s.itertuples(), hyp, ntok):
        mw = metrics.wer_cer([r.transcription], [h])
        rows.append({"seg_id": r.seg_id, "start_time": r.start_time, "duration": r.duration,
                     "mots_ref": len(r.transcription.split()), "mots_ft": len(h.split()), "tokens_ft": k,
                     "WER_ft": round(mw["WER"], 1), "repetition_ft": round(repetition(h), 2),
                     "limite_tokens": k >= MAX_NEW_TOKENS - 5, "ref": r.transcription, "hyp_ft": h})
    seg = pd.DataFrame(rows)
    seg["boucle"] = (seg.mots_ft > 2 * seg.mots_ref) & (seg.repetition_ft >= 0.3)
    errs = (seg.WER_ft / 100 * seg.mots_ref)
    print(f"   segments en boucle (sortie > 2 x la référence et >= 30 % de trigrammes répétés) : {seg.boucle.sum()} / "
          f"{len(seg)} ; ils portent {errs[seg.boucle].sum():.0f} des {errs.sum():.0f} erreurs")
    print(f"   segments à la limite de {MAX_NEW_TOKENS} tokens : {seg.limite_tokens.sum()}")
    ok = ~seg.boucle
    if ok.any():
        m_ok = metrics.wer_cer(seg.ref[ok].tolist(), seg.hyp_ft[ok].tolist())
        print(f"   WER sans les segments en boucle : {m_ok['WER']:.1f} % ({ok.sum()} segments)")

    # 4) zero-shot et décalage, sur le fichier audio complet du talk
    full, sr = sf.read(str(wav_path), dtype="float32")
    assert sr == SAMPLING_RATE
    off0 = float(data.talk_offsets().get(T, 0.0))
    tr = Transcriber(a.zs_model, 8)
    grid = np.round(np.arange(-a.max_offset, a.max_offset + 1e-9, a.step), 2)
    grows = []
    segs = s.sort_values("start_time")
    for d in grid:
        for r, h in zip(segs.itertuples(), tr(clips(full, segs, off0 + d))):
            grows.append({"seg_id": r.seg_id, "start_time": r.start_time, "decalage_s": float(d),
                          "CER_zs": seg_cer(r.transcription, h), "hyp_zs": h})
    gr = pd.DataFrame(grows)
    by_off = gr.groupby("decalage_s").CER_zs.mean()
    best = gr.loc[gr.groupby("seg_id").CER_zs.idxmin()].set_index("seg_id")
    at0 = gr[gr.decalage_s == 0].set_index("seg_id")
    seg = seg.merge(at0[["CER_zs"]].rename(columns={"CER_zs": "CER_zs_0"}), left_on="seg_id", right_index=True)
    seg = seg.merge(best[["decalage_s", "CER_zs"]].rename(columns={"decalage_s": "meilleur_decalage_s",
                                                                   "CER_zs": "CER_zs_meilleur"}),
                    left_on="seg_id", right_index=True)
    b_off = float(by_off.idxmin())
    share = float(np.mean(np.abs(seg.meilleur_decalage_s - b_off) <= 1.0))
    print(f"== 3. Zero-shot ({a.zs_model}), décalage actuel {off0:+.1f} s")
    print(f"   CER moyen à 0 s : {by_off.loc[0.0]:.1f} % ; meilleur décalage commun : {b_off:+.1f} s "
          f"(CER {by_off.min():.1f} %) ; segments dont le meilleur décalage est à ±1 s de celui-ci : {100 * share:.0f} %")
    print(f"   gain moyen par segment au meilleur décalage propre : "
          f"{(seg.CER_zs_0 - seg.CER_zs_meilleur).mean():.1f} points")

    # 5) 3 segments (pires WER du fine-tuné) : durée et énergie
    print("== 4. Durée et énergie des 3 segments au pire WER")
    worst = seg.sort_values("WER_ft", ascending=False).head(3)
    e_rows = []
    for r in worst.itertuples():
        p = s.loc[s.seg_id == r.seg_id, "path"].iloc[0]
        e_rows.append({"seg_id": r.seg_id, "duree_annotee_s": round(r.duration, 3), **energy(p),
                       "car_par_s_ref": round(len(r.ref) / r.duration, 1), "mots_ref": r.mots_ref,
                       "mots_ft": r.mots_ft, "WER_ft": r.WER_ft, "CER_zs_0": r.CER_zs_0})
    E = pd.DataFrame(e_rows)
    print(E.to_string(index=False))
    seg = seg.merge(E.drop(columns=["mots_ref", "mots_ft", "WER_ft", "CER_zs_0"]), on="seg_id", how="left")
    out = RESULTS_DIR / f"talk_check_{short}.csv"
    seg.sort_values("start_time").to_csv(out, index=False)
    gr.to_csv(RESULTS_DIR / f"talk_check_{short}_offsets.csv", index=False)
    print(f"-> {out}")
    if a.wide:
        wide_search(a, T, short, g_all, full, off0)


def confirm(a):
    """Contrôle indépendant du modèle fine-tuné (qui a vu ces textes) : Whisper zero-shot sur les 40 segments de
    l'échantillon du notebook 01, au décalage actuel et au décalage trouvé."""
    import soundfile as sf
    T = a.talk
    short = "_".join(T.split("_")[:2])
    tc = pd.read_parquet(DATA_DIR / "train_clean.parquet")
    s = tc[tc.talk == T]
    s = s.sample(min(a.n, len(s)), random_state=SEED).sort_values("start_time")
    full, _ = sf.read(str(AUDIO_DIR / f"{T}.wav"), dtype="float32")
    off0 = float(data.talk_offsets().get(T, 0.0))
    tr = Transcriber(a.zs_model, 8)
    rows = []
    for d in (0.0, a.confirm):
        for r, h in zip(s.itertuples(), tr(clips(full, s, off0 + d))):
            rows.append({"seg_id": r.seg_id, "decalage_s": d, "CER_zs": seg_cer(r.transcription, h), "hyp_zs": h})
    c = pd.DataFrame(rows)
    c.to_csv(RESULTS_DIR / f"talk_check_{short}_confirm.csv", index=False)
    p = c.pivot(index="seg_id", columns="decalage_s", values="CER_zs")
    better = float(np.mean(p[a.confirm] < p[0.0]))
    print(f"== 6. Contrôle zero-shot ({a.zs_model}, {len(s)} segments) : CER moyen {p[0.0].mean():.1f} % à "
          f"{off0:+.1f} s, {p[a.confirm].mean():.1f} % à {off0 + a.confirm:+.1f} s ; meilleur au décalage trouvé pour "
          f"{100 * better:.0f} % des segments (médiane talks du contrôle d'alignement : 58,4 %)")


def wide_search(a, T, short, g_all, full, off0):
    """Recherche large : où, dans l'audio, se trouve le texte de chaque segment ? Le modèle fine-tuné transcrit le
    segment décalé de -WIDE à +WIDE s ; le décalage au CER minimal, s'il donne un creux net, dit où le texte est
    prononcé. Concordant pour tous les segments = décalage constant ; croissant avec le temps = dérive (autre
    montage ou vitesse) ; sans creux = texte introuvable dans cet audio."""
    import torch
    from common import asr as A
    c = g_all[(g_all.duration.between(3, 20)) & (g_all.n_chars >= 15)].sort_values("start_time").reset_index(drop=True)
    segs = c.iloc[np.unique(np.linspace(0, len(c) - 1, min(a.wide_n, len(c))).round().astype(int))]
    grid = np.round(np.arange(-a.wide, a.wide + 1e-9, a.wide_step), 2)
    proc, model = A.load_whisper(a.ft_model, for_training=False)
    if torch.cuda.is_available():
        model = model.half()
    rows = []
    for d in grid:
        hyps = A.whisper_generate(model, proc, clips(full, segs, off0 + d), batch_size=16, max_new_tokens=96)
        for r, h in zip(segs.itertuples(), hyps):
            rows.append({"seg_id": r.seg_id, "start_time": r.start_time, "decalage_s": float(d),
                         "CER_ft": seg_cer(r.transcription, h), "hyp_ft": h})
    del model
    torch.cuda.empty_cache()
    w = pd.DataFrame(rows)
    w.to_csv(RESULTS_DIR / f"talk_check_{short}_wide.csv", index=False)
    summ = []
    for sid, g in w.groupby("seg_id"):
        g = g.sort_values("CER_ft")
        b = g.iloc[0]
        far = g[(g.decalage_s - b.decalage_s).abs() > 5]   # meilleur CER hors du voisinage du creux
        summ.append({"seg_id": sid, "start_time": b.start_time, "meilleur_decalage_s": b.decalage_s,
                     "CER_meilleur": round(b.CER_ft, 1), "CER_a_0": round(float(g[g.decalage_s == 0].CER_ft.iloc[0]), 1),
                     "CER_median_grille": round(float(g.CER_ft.median()), 1),
                     "CER_hors_creux": round(float(far.CER_ft.min()), 1) if len(far) else np.nan})
    S = pd.DataFrame(summ).sort_values("start_time")
    S["creux_net"] = (S.CER_hors_creux - S.CER_meilleur) >= 20
    S.to_csv(RESULTS_DIR / f"talk_check_{short}_wide_summary.csv", index=False)
    print(f"== 5. Recherche large (modèle fine-tuné, {len(segs)} segments, {grid[0]:+.0f} à {grid[-1]:+.0f} s, "
          f"pas {a.wide_step} s)")
    print(S.to_string(index=False))
    net = S[S.creux_net]
    if len(net) >= 3:
        slope, icpt = np.polyfit(net.start_time / 60, net.meilleur_decalage_s, 1)
        print(f"   creux nets : {len(net)} / {len(S)} ; décalages {net.meilleur_decalage_s.min():+.0f} à "
              f"{net.meilleur_decalage_s.max():+.0f} s ; pente {slope:+.2f} s par minute de talk")
    else:
        print(f"   creux nets : {len(net)} / {len(S)}")


if __name__ == "__main__":
    main()
