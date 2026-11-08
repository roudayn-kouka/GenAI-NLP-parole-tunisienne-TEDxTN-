"""Prépare le corpus en une commande (équivalent des premières cellules du notebook 01).

    python scripts/prepare_data.py              # annotations + audio + segments
    python scripts/prepare_data.py --no-audio   # annotations seulement (exploration textuelle)

Étapes : téléchargement des CSV du Hub -> contrôle des splits et des fuites ->
URL YouTube -> audio WAV 16 kHz -> découpage des segments -> workdir/data/corpus.parquet
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import data  # noqa: E402
from common.config import DATA_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-search", action="store_true", help="ne pas chercher les URL manquantes sur YouTube")
    ap.add_argument("--check-urls", action="store_true",
                    help="construire et contrôler youtube_urls.csv (durée, chaîne, titre) sans rien télécharger")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    df = data.load_corpus()
    print(data.check_splits(df).to_string(index=False))
    leak = data.check_leakage(df)
    print("Contrôle des fuites :", leak)
    if not leak["ok"]:
        sys.exit("Fuite détectée entre splits : arrêt.")
    if a.check_urls:
        urls = data.build_url_map(df, use_search=not a.no_search)
        print(urls.statut.value_counts().to_string())
        cols = ["audio_filename", "source", "duration_s", "derniere_annotation_s", "channel", "title", "motifs"]
        bad = urls[urls.statut != "OK"]
        if len(bad):
            print(bad[cols].to_string(index=False))
        print(f"-> {data.URL_MAP_FILE}")
        return
    if not a.no_audio:
        if a.no_search:
            data.build_url_map(df, use_search=False)
        rep = data.fetch_all_audio(df, workers=a.workers)
        rep.to_csv(DATA_DIR / "audio_report.csv", index=False)
        df = data.cut_segments(df)
    out = DATA_DIR / "corpus.parquet"
    df.to_parquet(out, index=False)
    print(f"Corpus prêt : {out} ({len(df)} segments)")


if __name__ == "__main__":
    main()
