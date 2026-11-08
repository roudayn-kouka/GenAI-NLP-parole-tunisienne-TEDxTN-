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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (AUDIO_DIR, DATA_DIR, HF_DATASET_ID, PAPER_SPLIT_STATS, RESULTS_DIR, SAMPLING_RATE,
                     SEGMENTS_DIR, SPLIT_FILES)
from .text import cmi, segment_type

URL_MAP_FILE = DATA_DIR / "youtube_urls.csv"   # audio_filename,url,source,duration_s
YOUTUBE_RE = re.compile(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}[^\s,\"']*)")
# TEDxTN_URLS.txt contient aussi des vidéos Facebook (2 talks de TEDxCarthage) : toute URL vidéo est acceptée,
# yt-dlp sait les télécharger.
VIDEO_URL_RE = re.compile(r"(https?://(?:www\.|m\.)?(?:youtube\.com/watch\?v=[\w\-]{11}|youtu\.be/[\w\-]{11}|"
                          r"facebook\.com/\S+|fb\.watch/\S+|vimeo\.com/\S+|dailymotion\.com/\S+)[^\s,\"']*)")


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
            m = VIDEO_URL_RE.search(line)
            if not m:
                continue
            # Le nom du talk est la première colonne non-URL de la ligne. TEDxTN_URLS.txt sépare
            # « URL nom_du_fichier » par un espace ou une tabulation : on découpe donc aussi sur les blancs.
            rest = VIDEO_URL_RE.sub(" ", line)
            parts = [x for x in re.split(r"[\s,;|]+", rest) if x and "http" not in x]
            name = parts[0] if parts else ""
            rows.append({"audio_filename": name, "url": m.group(1), "source": f"repo:{f}"})
    return pd.DataFrame(rows, columns=["audio_filename", "url", "source"])


class YouTubeBlocked(RuntimeError):
    """YouTube refuse temporairement l'accès (429 / vérification anti-robot) : rien à voir avec l'URL."""


_BLOCK_MARKERS = ("confirm you", "HTTP Error 429", "Too Many Requests")
_block_lock = threading.Lock()
_blocked_until = [0.0]   # partagé entre threads : un blocage met en pause toutes les requêtes


def _env(name: str, default: str) -> str:
    import os
    return os.environ.get(name, default)


def yt_max_parallel() -> int:
    """Nombre maximal de requêtes yt-dlp simultanées (YTDLP_MAX_PARALLEL, défaut 2)."""
    return max(1, int(_env("YTDLP_MAX_PARALLEL", "2")))


def _yt_dlp(args: list[str]) -> subprocess.CompletedProcess:
    """Appelle yt-dlp avec des options anti-blocage configurables par variables d'environnement :

    YTDLP_COOKIES             fichier de cookies Netscape (YouTube bloque les IP qui enchaînent les requêtes :
                              HTTP 429 puis « Sign in to confirm you're not a bot », y compris sur Colab)
    YTDLP_SLEEP_REQUESTS      pause entre requêtes HTTP d'une même extraction (s, défaut 1)
    YTDLP_SLEEP_INTERVAL      pause minimale avant chaque téléchargement (s, défaut 3)
    YTDLP_MAX_SLEEP_INTERVAL  pause maximale (tirage aléatoire, s, défaut 8)
    YTDLP_BLOCK_WAIT          attente après un blocage avant de reprendre la même requête (s, défaut 900)
    YTDLP_BLOCK_RETRIES       nombre de reprises après blocage (défaut 4)

    En cas de blocage, toutes les requêtes en cours attendent (pause partagée), puis reprennent là où elles
    s'étaient arrêtées (les téléchargements partiels .part sont repris par yt-dlp)."""
    extra = ["--cookies", _env("YTDLP_COOKIES", "")] if _env("YTDLP_COOKIES", "") else []
    extra += ["--sleep-requests", _env("YTDLP_SLEEP_REQUESTS", "1"),
              "--sleep-interval", _env("YTDLP_SLEEP_INTERVAL", "3"),
              "--max-sleep-interval", _env("YTDLP_MAX_SLEEP_INTERVAL", "8")]
    wait, retries = float(_env("YTDLP_BLOCK_WAIT", "900")), int(_env("YTDLP_BLOCK_RETRIES", "4"))
    for attempt in range(retries + 1):
        with _block_lock:
            pause = _blocked_until[0] - time.time()
        if pause > 0:
            time.sleep(pause)
        r = subprocess.run(["yt-dlp", *extra, *args], capture_output=True, text=True)
        if r.returncode == 0 or not any(m in r.stderr for m in _BLOCK_MARKERS):
            return r
        if attempt < retries:
            with _block_lock:
                if _blocked_until[0] < time.time():
                    _blocked_until[0] = time.time() + wait
                    print(f"[yt-dlp] accès bloqué par YouTube : pause de {wait / 60:.0f} min "
                          f"(reprise {attempt + 1}/{retries}) - {time.strftime('%H:%M:%S')}", flush=True)
    return r
_YT_FIELDS = "%(webpage_url)s\t%(duration)s\t%(channel)s\t%(uploader)s\t%(title)s"


def _parse_info(line: str) -> dict:
    url, dur, channel, uploader, title = (line.split("\t") + [""] * 5)[:5]
    return {"url": url, "duration_s": float(dur) if dur not in ("NA", "") else None,
            "channel": channel if channel != "NA" else uploader, "title": title}


def video_info(url: str) -> dict | None:
    """Métadonnées d'une vidéo (durée, chaîne, titre) sans téléchargement.
    Renvoie None si la vidéo n'existe pas ; lève YouTubeBlocked si l'accès reste refusé après les reprises."""
    r = _yt_dlp(["--print", _YT_FIELDS, "--skip-download", "--no-playlist", url])
    if r.returncode == 0 and r.stdout.strip():
        return _parse_info(r.stdout.strip().splitlines()[0])
    if any(m in r.stderr for m in _BLOCK_MARKERS):
        raise YouTubeBlocked(r.stderr.strip().splitlines()[-1][:200])
    return None


def _safe_video_info(url: str):
    try:
        return video_info(url)
    except YouTubeBlocked as e:
        return e


def search_youtube(talk: str, n: int = 5) -> list[dict]:
    """Dernier recours : recherche YouTube par titre (les noms de fichiers reprennent le titre du talk).
    Renvoie les n premiers candidats avec durée, chaîne et titre, pour pouvoir choisir le bon."""
    query = talk.replace("_", " ")
    r = _yt_dlp(["--print", _YT_FIELDS, "--skip-download", f"ytsearch{n}:{query}"])
    if r.returncode != 0:
        return []
    return [_parse_info(l) for l in r.stdout.strip().splitlines() if l.strip()]


def _tokens(s: str) -> list[str]:
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def check_video(talk: str, info: dict | None, last_end: float | None) -> dict:
    """Compare une vidéo au talk attendu (nom de fichier « Titre_Prenom_Nom_TEDxEvent »).
    Statut : OK / douteux / introuvable, avec les motifs du doute."""
    if not info:
        return {"statut": "introuvable", "motifs": "aucune vidéo", "sim_titre": None}
    event = talk_event(talk)
    title_tok = set(_tokens(info.get("title", "")))
    title_cat = "".join(_tokens(info.get("title", "")))   # « TEDxPasha Street » -> « tedxpashastreet »
    talk_tok = _tokens(talk)
    ev_tok = _tokens(event)[0] if event != "inconnu" else None
    # Les noms de fichiers collent parfois des mots (« Raja_FarhatatTEDxCarthage », « SoumayaAtoui ») :
    # un mot est retrouvé s'il figure dans le titre, en mot isolé ou dans le titre sans espaces.
    found = lambda t: t in title_tok or (len(t) >= 4 and t in title_cat)  # noqa: E731
    words = [t.replace(ev_tok, "") if ev_tok and ev_tok in t else t for t in talk_tok if t != ev_tok]
    words = [t for t in words if t and t not in {"at", "tedx"}]
    content = [t for t in words if len(t) >= 3]
    sim = sum(found(t) for t in content) / max(1, len(content))
    speaker = words[-2:]                     # les deux mots avant l'événement : en général le nom de l'orateur
    speaker_ok = all(found(t) for t in speaker)
    channel = info.get("channel") or ""
    motifs = []
    if last_end and (info.get("duration_s") is None or info["duration_s"] + 1.0 < last_end):
        motifs.append(f"vidéo ({info.get('duration_s')} s) plus courte que la dernière annotation ({last_end:.0f} s)")
    if "tedx" not in channel.lower():
        motifs.append(f"chaîne non TEDx ({channel or 'inconnue'})")
    if ev_tok and ev_tok not in title_cat:
        motifs.append(f"événement {event} absent du titre")
    if speaker and not speaker_ok:
        motifs.append(f"orateur ({' '.join(speaker)}) absent du titre")
        if sim < 0.5:   # titre traduit (arabe) mais orateur présent : pas un motif de doute à lui seul
            motifs.append(f"titre peu ressemblant (sim={sim:.2f})")
    return {"statut": "douteux" if motifs else "OK", "motifs": " ; ".join(motifs), "sim_titre": round(sim, 2)}


def _score(c: dict) -> tuple:
    return (c["statut"] == "OK", -len(c["motifs"].split(" ; ")) if c["motifs"] else 0, c["sim_titre"] or 0)


URL_MAP_COLUMNS = ["audio_filename", "url", "source", "duration_s", "channel", "title", "derniere_annotation_s",
                   "sim_titre", "statut", "motifs", "offset_s"]


def build_url_map(df: pd.DataFrame, use_search: bool = True, verify: bool = True) -> pd.DataFrame:
    """Construit (ou complète) DATA_DIR/youtube_urls.csv. Ce fichier peut être corrigé à la main :
    il est la source de vérité pour le téléchargement (une ligne « statut=exclu » n'est pas téléchargée ;
    « offset_s » décale le découpage de tous les segments du talk ; vider « statut » après avoir changé une URL
    pour qu'elle soit contrôlée à nouveau).

    Avec verify=True, chaque URL est contrôlée avant téléchargement (durée, chaîne TEDx, orateur et événement
    dans le titre). Pour les talks sans URL officielle, on garde le meilleur des 5 premiers résultats de recherche."""
    talks = sorted(df.talk.unique())
    last_end = df.groupby("talk").end_time.max()
    known = pd.read_csv(URL_MAP_FILE, keep_default_na=False) if URL_MAP_FILE.exists() else pd.DataFrame(
        columns=URL_MAP_COLUMNS)
    if known.empty:
        try:
            repo = discover_urls_in_repo()
            unmatched = sorted(set(repo.audio_filename) - set(talks))
            if unmatched:
                print(f"{len(unmatched)} noms du fichier d'URL absents des annotations : {unmatched[:5]}")
            known = repo[repo.audio_filename.isin(talks)].drop_duplicates("audio_filename")
        except Exception as e:  # dépôt inaccessible, pas de fichier d'URL, etc.
            print("Pas d'URL trouvées dans le dépôt :", e)
    known = known.reindex(columns=URL_MAP_COLUMNS)
    for c in ("url", "source", "channel", "title", "statut", "motifs"):
        known[c] = known[c].astype(object).where(known[c].notna(), "")
    for c in ("duration_s", "sim_titre"):
        known[c] = pd.to_numeric(known[c], errors="coerce").astype(object)
    known["offset_s"] = pd.to_numeric(known["offset_s"], errors="coerce").fillna(0.0)
    known["derniere_annotation_s"] = known.audio_filename.map(last_end)

    if verify:
        todo = known[(known.statut.fillna("") == "") & (known.url.fillna("") != "")]
        print(f"Contrôle de {len(todo)} URL (métadonnées YouTube)...")
        with ThreadPoolExecutor(yt_max_parallel()) as ex:   # peu de parallélisme : YouTube bloque vite (HTTP 429)
            infos = list(ex.map(_safe_video_info, todo.url))
        n_blocked = sum(isinstance(i, YouTubeBlocked) for i in infos)
        if n_blocked:
            print(f"ATTENTION : {n_blocked} URL non vérifiées, accès bloqué par YouTube ({infos[[isinstance(i, YouTubeBlocked) for i in infos].index(True)]}). "
                  "Elles restent sans statut : relancer plus tard ou avec YTDLP_COOKIES.")
        for idx, t, info in zip(todo.index, todo.audio_filename, infos):
            if isinstance(info, YouTubeBlocked):
                known.loc[idx, "motifs"] = "non vérifié : accès bloqué par YouTube"
                continue
            chk = check_video(t, info, last_end.get(t))
            if info:
                known.loc[idx, ["duration_s", "channel", "title"]] = [info["duration_s"], info["channel"], info["title"]]
            known.loc[idx, ["sim_titre", "statut", "motifs"]] = [chk["sim_titre"], chk["statut"], chk["motifs"]]

    missing = [t for t in talks if t not in set(known.audio_filename)]
    if use_search and missing:
        print(f"Recherche YouTube pour {len(missing)} talks sans URL...")
        with ThreadPoolExecutor(yt_max_parallel()) as ex:
            found = list(ex.map(search_youtube, missing))
        rows = []
        for t, cands in zip(missing, found):
            scored = [{**c, **check_video(t, c, last_end.get(t))} for c in cands]
            best = max(scored, key=_score) if scored else None
            if best is None:
                rows.append({"audio_filename": t, "url": "", "source": "ytsearch", "statut": "introuvable",
                             "motifs": "aucun résultat", "offset_s": 0.0, "derniere_annotation_s": last_end.get(t)})
            else:
                rows.append({"audio_filename": t, "source": "ytsearch (À VÉRIFIER)", "offset_s": 0.0,
                             "derniere_annotation_s": last_end.get(t),
                             **{k: best[k] for k in ("url", "duration_s", "channel", "title", "sim_titre", "statut", "motifs")}})
        known = pd.concat([known, pd.DataFrame(rows)], ignore_index=True)
    known = known.reindex(columns=URL_MAP_COLUMNS).sort_values("audio_filename")
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
        print(f"[échec] {talk}: {r.stderr.strip()[-300:]}", flush=True)
        return None
    print(f"[ok] {talk} ({out.stat().st_size / 1e6:.0f} Mo) - {time.strftime('%H:%M:%S')}", flush=True)
    return out


def fetch_all_audio(df: pd.DataFrame, workers: int = 2) -> pd.DataFrame:
    """Télécharge tous les talks et vérifie que la durée de l'audio couvre les annotations.
    Une durée trop courte signale une mauvaise vidéo (recherche par titre erronée).
    Reprise : les talks déjà téléchargés sont sautés ; relancer après une interruption suffit."""
    import soundfile as sf
    urls = build_url_map(df)
    todo = urls[(urls.url.fillna("") != "") & ~urls.statut.isin(["introuvable", "exclu"])]
    workers = min(workers, yt_max_parallel())
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(lambda r: download_talk(r.audio_filename, r.url), todo.itertuples()))
    last_end = df.groupby("talk").end_time.max()
    offsets = talk_offsets()
    report = []
    for (t, p, st) in zip(todo.audio_filename, paths, todo.statut):
        dur = sf.info(str(p)).duration if p else None
        need = last_end.get(t, 0) + float(offsets.get(t, 0.0))
        ok = dur is not None and dur + 1.0 >= need
        report.append({"talk": t, "audio_s": dur, "derniere_annotation_s": last_end.get(t),
                       "offset_s": offsets.get(t, 0.0), "statut_url": st, "coherent": ok})
    rep = pd.DataFrame(report)
    missing = sorted(set(df.talk) - set(rep[rep.coherent].talk))
    print(f"{rep.coherent.sum()} talks OK / {df.talk.nunique()} ; à vérifier : {len(missing)}")
    return rep


def talk_offsets() -> dict[str, float]:
    """Décalage temporel (s) par talk, lu dans youtube_urls.csv (colonne offset_s) : la vidéo YouTube peut
    contenir une intro absente de l'audio annoté. Temps audio = temps annoté + offset_s."""
    if not URL_MAP_FILE.exists():
        return {}
    m = pd.read_csv(URL_MAP_FILE, keep_default_na=False)
    if "offset_s" not in m:
        return {}
    off = pd.to_numeric(m.offset_s, errors="coerce").fillna(0.0)
    return dict(zip(m.audio_filename, off))


def excluded_talks() -> set[str]:
    if not URL_MAP_FILE.exists():
        return set()
    m = pd.read_csv(URL_MAP_FILE, keep_default_na=False)
    return set(m.loc[m.get("statut", pd.Series(dtype=str)).isin(["exclu", "introuvable"]), "audio_filename"])


def cut_segments(df: pd.DataFrame, out_dir: Path = SEGMENTS_DIR, pad_s: float = 0.0,
                 overwrite: bool = False) -> pd.DataFrame:
    """Découpe les segments annotés dans l'audio complet et ajoute la colonne `path`.
    Applique le décalage `offset_s` de youtube_urls.csv. Les segments dont le talk n'est pas disponible
    ou est exclu (statut exclu / introuvable) sont retirés (et comptés).
    Le décalage est inscrit dans le nom des fichiers, si bien qu'un changement d'offset_s force le redécoupage."""
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    offsets, excluded = talk_offsets(), excluded_talks()
    paths = []
    cache: dict[str, tuple[np.ndarray, int]] = {}
    for row in df.sort_values(["talk", "start_time"]).itertuples():
        off = float(offsets.get(row.talk, 0.0))
        dst = out_dir / (f"{row.seg_id}.wav" if off == 0 else f"{row.seg_id}__off{off:+.1f}.wav")
        if row.talk in excluded:
            paths.append((row.Index, None))
            continue
        if overwrite or not dst.exists():
            src = AUDIO_DIR / f"{row.talk}.wav"
            if not src.exists():
                paths.append((row.Index, None))
                continue
            if row.talk not in cache:
                cache.clear()  # garde un seul talk en mémoire
                cache[row.talk] = sf.read(str(src), dtype="float32")
            wav, sr = cache[row.talk]
            a = max(0, int((row.start_time + off - pad_s) * sr))
            b = min(len(wav), int((row.end_time + off + pad_s) * sr))
            if b - a <= 0:
                paths.append((row.Index, None))
                continue
            sf.write(str(dst), wav[a:b], sr)
        paths.append((row.Index, str(dst)))
    out = df.copy()
    out["path"] = pd.Series(dict(paths))
    n_missing = out.path.isna().sum()
    if n_missing:
        print(f"{n_missing} segments sans audio (talks non téléchargés ou exclus) retirés.")
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
