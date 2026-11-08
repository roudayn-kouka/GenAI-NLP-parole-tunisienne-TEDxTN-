"""Archive unique pour Colab (notebook 01b), sans compression (le WAV se compresse mal) :

    python scripts/make_colab_archive.py [--out workdir/tedxtn_segments.tar]

Contenu (chemins relatifs à TEDXTN_WORKDIR) :
    data/corpus.parquet                     corpus découpé et contrôlé (après scripts/check_alignment.py)
    audio_segments/*.wav                    uniquement les segments référencés par corpus.parquet
    models/asr_w2v-bert-2.0/vocab.json      vocabulaire CTC appris sur le train
    lm/tedxtn_4gram.bin                     LM KenLM 4-gram estimé sur le train uniquement
    lm/train_ctc.txt                        corpus du LM (train) : unigrammes du décodeur pyctcdecode

L'archive contient de l'audio TEDxTN (CC BY-NC-ND) : à garder sur un Drive privé, jamais publiée.
"""
import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import asr as A, data, text  # noqa: E402
from common.config import DATA_DIR, LM_DIR, MODELS_DIR, PROJECT_ROOT, TrainConfig, WORKDIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORKDIR / "tedxtn_segments.tar"))
    a = ap.parse_args()

    df = pd.read_parquet(DATA_DIR / "corpus.parquet")
    cfg = TrainConfig()
    train_df, _ = data.clean_for_training(df[df.split == "train"], cfg.min_audio_seconds, cfg.max_audio_seconds)

    vocab_dir = MODELS_DIR / "asr_w2v-bert-2.0"
    vocab = A.build_ctc_vocab(train_df.transcription.tolist(), vocab_dir)
    print(f"vocabulaire CTC : {len(vocab)} symboles")

    lm_bin = LM_DIR / "tedxtn_4gram.bin"
    corpus = LM_DIR / "train_ctc.txt"   # même texte que le notebook 01 §7 (train uniquement) ; sert aussi d'unigrammes
    corpus.write_text("\n".join(text.normalize_text(t, A.CTC_NORM) for t in train_df.transcription), encoding="utf-8")
    if not lm_bin.exists():
        subprocess.run(["bash", str(PROJECT_ROOT / "scripts" / "build_kenlm.sh"), str(corpus), str(LM_DIR), "4"], check=True)

    files = [DATA_DIR / "corpus.parquet", vocab_dir / "vocab.json", lm_bin, corpus] + [Path(p) for p in df.path]
    missing = [f for f in files if not f.exists()]
    assert not missing, f"{len(missing)} fichiers manquants, ex. {missing[:3]}"
    out = Path(a.out)
    with tarfile.open(out, "w") as tar:     # « w » = sans compression
        for f in files:
            arc = f"audio_segments/{f.name}" if f.suffix == ".wav" else str(f.relative_to(WORKDIR))
            tar.add(str(f), arcname=arc)
    print(f"{out} : {len(files)} fichiers, {out.stat().st_size / 1e9:.2f} Go "
          f"({len(df)} segments ; train {df[df.split == 'train'].duration.sum() / 3600:.2f} h)")


if __name__ == "__main__":
    main()
