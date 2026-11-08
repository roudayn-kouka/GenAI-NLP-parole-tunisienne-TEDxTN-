"""
Accès au corpus TEDxTN : téléchargement des annotations, récupération de l'audio,
découpage en segments, contrôle des fuites de données et construction des datasets.

Format vérifié du dépôt Hugging Face `fbougares/TEDxTN` :
    train.csv / dev.csv / test.csv          audio_filename,start_time,end_time,transcription
    train.en.csv / dev.en.csv / test.en.csv  audio_filename,start_time,end_time,translation

L'audio n'est PAS redistribué (licence CC BY-NC-ND 4.0) : le papier fournit l'URL
YouTube de chaque talk. `fetch_all_audio()` cherche ces URL dans le dépôt, puis
utilise une recherche YouTube par titre en dernier recours (à vérifier, cf. README).
"""
from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (AUDIO_DIR, DATA_DIR, HF_DATASET_ID, PAPER_SPLIT_STATS, RESULTS_DIR, SAMPLING_RATE,
                     SEGMENTS_DIR, SPLIT_FILES)
from .text import cmi, segment_type

URL_MAP_FILE = DATA_DIR / "youtube_urls.csv"   # audio_filename,url,source,duration_s
YOUTUBE_RE = re.compile(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}[^\s,\"']*)")


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------
def download_annotations(force: bool = False, token: str | None = None) -> dict[str, Path]:
    """Télécharge les 6 CSV depuis le Hub dans DATA_DIR. Renvoie {nom_fichier: chemin}."""
    from huggingface_hub import hf_hub_download
    paths = {}
    for split, files in SPLIT_FILES.items():
        for fname in files:
            local = DATA_DIR / fname
            if force or not local.exists():
                p = hf_hub_download(HF_DATASET_ID, fname, repo_type="dataset", local_dir=DATA_DIR, token=token)
                local = Path(p)
            paths[fname] = local
    return paths


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"audio_filename": str}, keep_default_na=False)
    df["start_time"] = df["start_time"].astype(float).round(3)
    df["end_time"] = df["end_time"].astype(float).round(3)
    return df


def load_split(split: str) -> pd.DataFrame:
    """Charge un split et fusionne transcription + traduction sur (talk, début, fin)."""
    tr_file, en_file = SPLIT_FILES[split]
    tr = _read_csv(DATA_DIR / tr_file)
    key = ["audio_filename", "start_time", "end_time"]
    en_path = DATA_DIR / en_file
    if en_path.exists():
        en = _read_csv(en_path)
        df = tr.merge(en[key + ["translation"]], on=key, how="left")
    else:
        df = tr.assign(translation="")
    df["translation"] = df["translation"].fillna("")
    df.insert(0, "split", split)
    return df


def talk_event(audio_filename: str) -> str:
    """Extrait l'événement TEDx du nom de fichier (…_TEDxNabeul -> TEDxNabeul)."""
    m = re.search(r"(TEDx[A-Za-z0-9]+)\s*$", audio_filename)
    return m.group(1) if m else "inconnu"


def load_corpus(download: bool = True) -> pd.DataFrame:
    """Corpus complet avec colonnes dérivées :
    seg_id, talk, event, duration, n_words, n_chars, seg_type (TUN/MIXED/FOR), cmi."""
    if download:
        download_annotations()
    df = pd.concat([load_split(s) for s in SPLIT_FILES], ignore_index=True)
    df["talk"] = df["audio_filename"]
    df["event"] = df["talk"].map(talk_event)
    df["duration"] = (df["end_time"] - df["start_time"]).round(3)
    df["n_words"] = df["transcription"].str.split().str.len().fillna(0).astype(int)
    df["n_chars"] = df["transcription"].str.len()
    df["n_words_en"] = df["translation"].str.split().str.len().fillna(0).astype(int)
    df["seg_type"] = df["transcription"].map(segment_type)
    df["cmi"] = df["transcription"].map(cmi)
    df["seg_id"] = (df["split"] + "__" + df.groupby("split").cumcount().astype(str).str.zfill(5))
    return df


# --------------------------------------------------------------------------
# Contrôles de qualité et de fuite
# --------------------------------------------------------------------------
def check_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Compare les splits chargés aux chiffres du papier (Table 5)."""
    rows = []
    for split, ref in PAPER_SPLIT_STATS.items():
        d = df[df.split == split]
        rows.append({"split": split, "talks": d.talk.nunique(), "talks_papier": ref["talks"],
                     "segments": len(d), "segments_papier": ref["segments"],
                     "heures": round(d.duration.sum() / 3600, 2)})
    return pd.DataFrame(rows)


def check_leakage(df: pd.DataFrame) -> dict:
    """Vérifie l'absence de fuite entre splits :
    1) aucun talk (donc aucun locuteur principal) partagé ;
    2) aucune transcription identique longue partagée (texte dupliqué) ;
    3) aucun chevauchement temporel de segments dans un même talk."""
    talks = {s: set(df[df.split == s].talk) for s in df.split.unique()}
    shared_talks = {f"{a}∩{b}": sorted(talks[a] & talks[b])
                    for a in talks for b in talks if a < b and talks[a] & talks[b]}
    long = df[df.n_words >= 6]
    texts = {s: set(long[long.split == s].transcription) for s in df.split.unique()}
    shared_texts = {f"{a}∩{b}": len(texts[a] & texts[b]) for a in texts for b in texts if a < b}
    overlaps = 0
    for _, g in df.sort_values(["talk", "start_time"]).groupby("talk"):
        overlaps += int((g.start_time.values[1:] < g.end_time.values[:-1] - 1e-3).sum())
    return {"talks_partages": shared_talks, "transcriptions_partagees(>=6 mots)": shared_texts,
            "segments_qui_se_chevauchent": overlaps,
            "ok": not shared_talks}


def clean_for_training(df: pd.DataFrame, min_s: float = 0.3, max_s: float = 30.0,
                       max_chars_per_s: float = 40.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtre les segments inutilisables pour l'entraînement (le dev/test n'est jamais filtré).
    Renvoie (gardés, rejetés avec la raison)."""
    reasons = pd.Series("", index=df.index)
    reasons[df.transcription.str.strip() == ""] = "transcription vide"
    reasons[(reasons == "") & (df.duration < min_s)] = f"durée < {min_s}s"
    reasons[(reasons == "") & (df.duration > max_s)] = f"durée > {max_s}s (fenêtre Whisper)"
    rate = df.n_chars / df.duration.clip(lower=1e-3)
    reasons[(reasons == "") & (rate > max_chars_per_s)] = "débit irréaliste (alignement douteux)"
    rej = df[reasons != ""].assign(raison=reasons[reasons != ""])
    return df[reasons == ""].copy(), rej


# --------------------------------------------------------------------------
# Audio : URL YouTube, téléchargement, découpage
# --------------------------------------------------------------------------
def discover_urls_in_repo(token: str | None = None) -> pd.DataFrame:
    """Cherche dans les fichiers texte du dépôt toute URL YouTube et l'associe au talk."""
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(HF_DATASET_ID, repo_type="dataset", token=token)
    candidates = [f for f in files if f.lower().endswith((".csv", ".tsv", ".txt", ".json", ".jsonl", ".md"))
                  and f not in {x for pair in SPLIT_FILES.values() for x in pair}]
    rows = []
    for f in candidates:
        p = hf_hub_download(HF_DATASET_ID, f, repo_type="dataset", local_dir=DATA_DIR, token=token)
        for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
            m = YOUTUBE_RE.search(line)
            if not m:
                continue
            # Le nom du talk est la première colonne non-URL de la ligne
            parts = [x.strip() for x in re.split(r"[,\t;|]", line) if x.strip() and "http" not in x]
            name = parts[0] if parts else ""
            rows.append({"audio_filename": name, "url": m.group(1), "source": f"repo:{f}"})
    return pd.DataFrame(rows, columns=["audio_filename", "url", "source"])


def _yt_dlp(args: list[str]) -> subprocess.CompletedProcess:
    """YouTube bloque parfois les IP de Colab (« confirm you're not a bot ») : on peut alors exporter
    les cookies de son navigateur (fichier Netscape) et indiquer leur chemin dans YTDLP_COOKIES."""
    import os
    extra = ["--cookies", os.environ["YTDLP_COOKIES"]] if os.environ.get("YTDLP_COOKIES") else []
    return subprocess.run(["yt-dlp", *extra, *args], capture_output=True, text=True)


def search_youtube(talk: str) -> tuple[str | None, float | None]:
    """Dernier recours : recherche YouTube par titre (les noms de fichiers reprennent le titre du talk)."""
    query = talk.replace("_", " ")
    r = _yt_dlp(["--print", "%(webpage_url)s\t%(duration)s", "--skip-download", f"ytsearch1:{query}"])
    if r.returncode != 0 or not r.stdout.strip():
        return None, None
    url, dur = r.stdout.strip().splitlines()[0].split("\t")
    return url, float(dur) if dur not in ("NA", "") else None


def build_url_map(df: pd.DataFrame, use_search: bool = True) -> pd.DataFrame:
    """Construit (ou complète) DATA_DIR/youtube_urls.csv. Ce fichier peut être corrigé à la main :
    il est la source de vérité pour le téléchargement."""
    talks = sorted(df.talk.unique())
    known = pd.read_csv(URL_MAP_FILE) if URL_MAP_FILE.exists() else pd.DataFrame(
        columns=["audio_filename", "url", "source", "duration_s"])
    if known.empty:
        try:
            repo = discover_urls_in_repo()
            known = repo[repo.audio_filename.isin(talks)].drop_duplicates("audio_filename")
        except Exception as e:  # dépôt inaccessible, pas de fichier d'URL, etc.
            print("Pas d'URL trouvées dans le dépôt :", e)
    missing = [t for t in talks if t not in set(known.audio_filename)]
    if use_search and missing:
        print(f"Recherche YouTube pour {len(missing)} talks sans URL...")
        with ThreadPoolExecutor(8) as ex:
            found = list(ex.map(search_youtube, missing))
        new = pd.DataFrame([{"audio_filename": t, "url": u, "source": "ytsearch (À VÉRIFIER)", "duration_s": d}
                            for t, (u, d) in zip(missing, found) if u])
        known = pd.concat([known, new], ignore_index=True)
    known.to_csv(URL_MAP_FILE, index=False)
    return known


def download_talk(talk: str, url: str, out_dir: Path = AUDIO_DIR) -> Path | None:
    """Télécharge l'audio d'un talk et le convertit en WAV mono 16 kHz."""
    out = out_dir / f"{talk}.wav"
    if out.exists() and out.stat().st_size > 0:
        return out
    tmpl = str(out_dir / f"{talk}.%(ext)s")
    r = _yt_dlp(["-f", "bestaudio/best", "-x", "--audio-format", "wav",
                 "--postprocessor-args", f"ffmpeg:-ac 1 -ar {SAMPLING_RATE}", "-o", tmpl, "--no-playlist", url])
    if r.returncode != 0 or not out.exists():
        print(f"[échec] {talk}: {r.stderr.strip()[-300:]}")
        return None
    return out


def fetch_all_audio(df: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """Télécharge tous les talks et vérifie que la durée de l'audio couvre les annotations.
    Une durée trop courte signale une mauvaise vidéo (recherche par titre erronée)."""
    import soundfile as sf
    urls = build_url_map(df)
    todo = urls.dropna(subset=["url"])
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(lambda r: download_talk(r.audio_filename, r.url), todo.itertuples()))
    last_end = df.groupby("talk").end_time.max()
    report = []
    for (t, p) in zip(todo.audio_filename, paths):
        dur = sf.info(str(p)).duration if p else None
        ok = dur is not None and dur + 1.0 >= last_end.get(t, 0)
        report.append({"talk": t, "audio_s": dur, "derniere_annotation_s": last_end.get(t), "coherent": ok})
    rep = pd.DataFrame(report)
    missing = sorted(set(df.talk) - set(rep[rep.coherent].talk))
    print(f"{rep.coherent.sum()} talks OK / {df.talk.nunique()} ; à vérifier : {len(missing)}")
    return rep


def cut_segments(df: pd.DataFrame, out_dir: Path = SEGMENTS_DIR, pad_s: float = 0.0) -> pd.DataFrame:
    """Découpe les segments annotés dans l'audio complet et ajoute la colonne `path`.
    Les segments dont le talk n'est pas disponible sont retirés (et comptés)."""
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for row in df.sort_values(["talk", "start_time"]).itertuples():
        dst = out_dir / f"{row.seg_id}.wav"
        if not dst.exists():
            src = AUDIO_DIR / f"{row.talk}.wav"
            if not src.exists():
                paths.append((row.Index, None))
                continue
            if row.talk not in cache:
                cache.clear()  # garde un seul talk en mémoire
                cache[row.talk] = sf.read(str(src), dtype="float32")
            wav, sr = cache[row.talk]
            a = max(0, int((row.start_time - pad_s) * sr))
            b = min(len(wav), int((row.end_time + pad_s) * sr))
            sf.write(str(dst), wav[a:b], sr)
        paths.append((row.Index, str(dst)))
    out = df.copy()
    out["path"] = pd.Series(dict(paths))
    n_missing = out.path.isna().sum()
    if n_missing:
        print(f"{n_missing} segments sans audio (talks non téléchargés) retirés.")
    return out.dropna(subset=["path"])


def load_audio(path_or_array, sr_in: int | None = None, sr: int = SAMPLING_RATE) -> np.ndarray:
    """Charge un fichier (ou un tableau numpy) en float32 mono rééchantillonné à `sr`."""
    import soundfile as sf
    if isinstance(path_or_array, (str, Path)):
        wav, sr_in = sf.read(str(path_or_array), dtype="float32", always_2d=False)
    else:
        wav = np.asarray(path_or_array)
        if np.issubdtype(wav.dtype, np.integer):
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        else:
            wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[1] <= 8 else 0)
    if sr_in and sr_in != sr:
        from scipy.signal import resample_poly
        g = np.gcd(int(sr_in), int(sr))
        wav = resample_poly(wav, sr // g, int(sr_in) // g).astype(np.float32)
    return wav


# --------------------------------------------------------------------------
# Datasets Hugging Face
# --------------------------------------------------------------------------
def to_hf_dataset(df: pd.DataFrame, text_col: str = "transcription"):
    """Construit un `datasets.Dataset` avec une colonne Audio décodée à 16 kHz."""
    from datasets import Audio, Dataset
    cols = ["seg_id", "talk", "path", "duration", "seg_type", "transcription", "translation"]
    ds = Dataset.from_pandas(df[cols].reset_index(drop=True))
    ds = ds.cast_column("path", Audio(sampling_rate=SAMPLING_RATE))
    return ds.rename_column("path", "audio")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def load_json(path: Path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else default


# --------------------------------------------------------------------------
# Prédictions sauvegardées (reprise après coupure Colab, travail en parallèle)
# --------------------------------------------------------------------------
PREDS_DIR = RESULTS_DIR / "preds"


def save_preds(task: str, system: str, split: str, seg_ids, hyps, refs=None, **extra_cols) -> Path:
    """Enregistre les sorties d'un système : results/preds/<task>__<system>__<split>.csv.
    Chaque membre de l'équipe peut produire ses fichiers ; les sections d'évaluation les rassemblent."""
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"seg_id": list(seg_ids), "hyp": list(hyps)})
    if refs is not None:
        df["ref"] = list(refs)
    for k, v in extra_cols.items():
        df[k] = list(v)
    path = PREDS_DIR / f"{task}__{system}__{split}.csv"
    df.to_csv(path, index=False)
    return path


def load_preds(task: str, split: str) -> dict[str, pd.DataFrame]:
    """{nom_du_système: DataFrame} pour toutes les prédictions disponibles d'une tâche et d'un split."""
    out = {}
    for p in sorted(PREDS_DIR.glob(f"{task}__*__{split}.csv")):
        out[p.stem.split("__")[1]] = pd.read_csv(p, keep_default_na=False)
    return out
