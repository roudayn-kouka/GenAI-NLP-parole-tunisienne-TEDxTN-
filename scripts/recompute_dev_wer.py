"""Recalcule les points de la courbe « WER dev (sous-ensemble) » perdus lors d'un arrêt de l'entraînement Whisper.

    python scripts/recompute_dev_wer.py                                  # models/asr_whisper-small
    python scripts/recompute_dev_wer.py --model-dir <dossier> --base whisper-small

Jusqu'à la correction de GenerationWERCallback (CHANGELOG_EXECUTION.md), les WER dev calculés à chaque évaluation
n'étaient gardés qu'en mémoire : un arrêt les perdait. Ce script recalcule, pour chaque checkpoint encore présent
dont le step n'a pas de point WER, la même mesure que le callback : mêmes 200 segments dev
(dev_df.sample(200, random_state=SEED)), poids fp32 du checkpoint, whisper_generate glouton, batch 4 (GPU < 12 Go)
ou 16. Les points sont ajoutés (avec "recalcule": true) au log_history du checkpoint le plus récent, celui depuis
lequel l'entraînement reprend : le callback les relit à la reprise et ils figurent dans la courbe finale.
Une copie trainer_state.json.avant_recalcul est gardée. Dev uniquement : le test n'est pas lu.
Les points des checkpoints déjà supprimés par la rotation (save_total_limit) ne sont pas récupérables.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import asr as A, data, metrics  # noqa: E402
from common.config import BASE_MODELS, MODELS_DIR, SEED, gpu_memory_gb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODELS_DIR / "asr_whisper-small"))
    ap.add_argument("--base", default="whisper-small")
    a = ap.parse_args()
    out = Path(a.model_dir)
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        print("aucun checkpoint dans", out); return
    last = ckpts[-1]
    state_f = last / "trainer_state.json"
    state = json.loads(state_f.read_text(encoding="utf-8"))
    done = {x["step"] for x in state["log_history"] if "dev_WER_subset" in x}
    todo = [c for c in ckpts if int(c.name.split("-")[1]) not in done]
    if not todo:
        print("courbe WER déjà complète pour les checkpoints présents :", sorted(done)); return

    # mêmes 200 segments dev que run_whisper_ft (notebook 01)
    df = data.cut_segments(data.load_corpus(download=False))
    dev_df = df[df.split == "dev"].reset_index(drop=True)
    sub = dev_df.sample(min(200, len(dev_df)), random_state=SEED)
    wavs = [data.load_audio(p) for p in sub.path]
    refs = sub.transcription.tolist()
    gen_bs = 16 if gpu_memory_gb() >= 12 else 4

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(BASE_MODELS[a.base])   # le checkpoint ne contient pas le tokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recs = []
    for c in todo:
        step = int(c.name.split("-")[1])
        model = WhisperForConditionalGeneration.from_pretrained(c).to(dev)   # fp32, comme pendant l'entraînement
        model.generation_config.language, model.generation_config.task = "arabic", "transcribe"   # = load_whisper
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        with torch.no_grad():
            hyps = A.whisper_generate(model, proc, wavs, batch_size=gen_bs)
        m = metrics.wer_cer(refs, hyps)
        recs.append({"step": step, "dev_WER_subset": m["WER"], "dev_CER_subset": m["CER"], "recalcule": True})
        print(f"[{c.name}] WER dev (sous-ensemble, recalculé) = {m['WER']:.2f}  CER = {m['CER']:.2f}")
        del model; torch.cuda.empty_cache()

    shutil.copy(state_f, state_f.with_name("trainer_state.json.avant_recalcul"))
    state["log_history"] = sorted(state["log_history"] + recs, key=lambda x: x["step"])
    state_f.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("points ajoutés au log_history de", last.name, ":", [r["step"] for r in recs])


if __name__ == "__main__":
    main()
