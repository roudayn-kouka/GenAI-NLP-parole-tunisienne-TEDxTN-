"""
Volet GenAI : un LLM instruct (par défaut Qwen2.5, multilingue et performant en arabe) utilisé pour
1) la post-correction des transcriptions ASR (few-shot, exemples tirés du train),
2) la traduction du texte reconnu (troisième variante du pipeline, comparée à NLLB),
3) la normalisation orthographique (notebook 03).

Garde-fou : si la sortie du LLM s'éloigne trop de l'entrée (ratio de longueur), on garde l'entrée.
Cela évite qu'une « correction » ne réécrive la phrase (hallucination).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_llm(model_id: str = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32}
    if load_in_4bit and DEVICE == "cuda":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                                       bnb_4bit_quant_type="nf4")
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if "device_map" not in kw:
        model = model.to(DEVICE)
    return tok, model.eval()


@torch.inference_mode()
def chat(tok, model, prompts: list[list[dict]], max_new_tokens: int = 200, batch_size: int = 8) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch_size):
        texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts[i:i + batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tok.pad_token_id)
        outs.extend(tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen)
    return outs


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip("«»").strip()
    s = re.sub(r"^(correction|corrected|translation|traduction)\s*:\s*", "", s, flags=re.I)
    return s.splitlines()[0].strip() if s else s


# --------------------------------------------------------------------------
# 1) Post-correction ASR
# --------------------------------------------------------------------------
CORRECTION_SYSTEM = (
    "You correct automatic transcriptions of Tunisian Arabic speech (Derja) with French/English code-switching. "
    "Rules: keep the same words and order; fix only obvious recognition errors (misheard words, broken words, "
    "wrong script for French/English loanwords: write them in Latin script). Never translate, never paraphrase, "
    "never add words. Output only the corrected transcription."
)


def correction_prompts(hyps: list[str], fewshot: list[tuple[str, str]]) -> list[list[dict]]:
    """fewshot : liste de (sortie ASR, référence) issus du train (jamais du test)."""
    prompts = []
    for h in hyps:
        msgs = [{"role": "system", "content": CORRECTION_SYSTEM}]
        for a, b in fewshot:
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": h})
        prompts.append(msgs)
    return prompts


def post_correct(tok, model, hyps: list[str], fewshot: list[tuple[str, str]], min_similarity: float = 0.6,
                 batch_size: int = 8) -> tuple[list[str], list[bool]]:
    """Renvoie (textes corrigés, drapeaux 'correction acceptée')."""
    raw = chat(tok, model, correction_prompts(hyps, fewshot), batch_size=batch_size)
    out, accepted = [], []
    for h, r in zip(hyps, raw):
        r = _clean(r)
        ok = bool(r) and SequenceMatcher(None, h, r).ratio() >= min_similarity
        out.append(r if ok else h)
        accepted.append(ok and r != h)
    return out, accepted


# --------------------------------------------------------------------------
# 2) Traduction par LLM
# --------------------------------------------------------------------------
def translation_prompts(texts: list[str], target: str, fewshot: list[tuple[str, str]] | None = None):
    system = (f"You are an expert translator of Tunisian Arabic (Derja), which often mixes French and English. "
              f"Translate the user's Tunisian sentence into fluent {target}. Translate the French/English parts too. "
              f"Keep disfluencies' meaning but not filler sounds. Output only the translation.")
    prompts = []
    for t in texts:
        msgs = [{"role": "system", "content": system}]
        for a, b in (fewshot or []):
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": t})
        prompts.append(msgs)
    return prompts


def llm_translate(tok, model, texts: list[str], target: str = "English", fewshot=None, batch_size: int = 8) -> list[str]:
    return [_clean(x) for x in chat(tok, model, translation_prompts(texts, target, fewshot), batch_size=batch_size)]


# --------------------------------------------------------------------------
# 3) Normalisation par LLM
# --------------------------------------------------------------------------
NORMALIZATION_SYSTEM = (
    "Rewrite the Tunisian Arabic transcription with a consistent spelling (CODA-Tunisian conventions): "
    "Arabic script for Tunisian words, plain alef (ا) instead of أ إ آ, ي instead of final ى, no diacritics, "
    "no letter elongation, French/English words in Latin script, separate the article ال from Latin words "
    "with a space. Do not change, add, remove or translate any word. Output only the rewritten text."
)


def llm_normalize(tok, model, texts: list[str], fewshot: list[tuple[str, str]] | None = None, batch_size: int = 8):
    prompts = []
    for t in texts:
        msgs = [{"role": "system", "content": NORMALIZATION_SYSTEM}]
        for a, b in (fewshot or []):
            msgs += [{"role": "user", "content": a}, {"role": "assistant", "content": b}]
        msgs.append({"role": "user", "content": t})
        prompts.append(msgs)
    return [_clean(x) for x in chat(tok, model, prompts, batch_size=batch_size)]
