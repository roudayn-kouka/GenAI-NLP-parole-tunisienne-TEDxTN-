"""
Traduction : texte tunisien -> anglais / français / arabe standard (pipeline en deux étapes),
et baselines de traduction de la parole (SeamlessM4T) ainsi qu'un modèle
encodeur-parole + décodeur-texte (diagnostic de l'échec rapporté par le papier).

NLLB-200 possède un code de langue dédié au tunisien (`aeb_Arab`), ce qui en fait
un choix naturel pour la seconde étape du pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import NLLB_SRC_LANG, SAMPLING_RATE, TARGET_LANGUAGES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# NLLB (texte -> texte)
# ==========================================================================
def load_nllb(model_id: str, src_lang: str = NLLB_SRC_LANG, fp16: bool = True):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)
    dtype = torch.float16 if (fp16 and DEVICE == "cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE).eval()
    return tok, model


@torch.inference_mode()
def nllb_translate(tok, model, texts: list[str], tgt_lang: str = "eng_Latn", batch_size: int = 32,
                   num_beams: int = 4, max_new_tokens: int = 256) -> list[str]:
    out = []
    bos = tok.convert_tokens_to_ids(tgt_lang)
    for i in range(0, len(texts), batch_size):
        chunk = [t if str(t).strip() else "." for t in texts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
        gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=num_beams, max_new_tokens=max_new_tokens)
        out.extend(x.strip() for x in tok.batch_decode(gen, skip_special_tokens=True))
    return out


class TextPairDataset(torch.utils.data.Dataset):
    """Paires (transcription tunisienne, traduction anglaise) pour fine-tuner NLLB."""

    def __init__(self, df: pd.DataFrame, tok, src_col="transcription", tgt_col="translation", max_len=256):
        self.src = df[src_col].astype(str).tolist()
        self.tgt = df[tgt_col].astype(str).tolist()
        self.tok, self.max_len = tok, max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        enc = self.tok(self.src[i], text_target=self.tgt[i], truncation=True, max_length=self.max_len)
        return enc


class Seq2SeqCollator:
    """DataCollatorForSeq2Seq + `decoder_input_ids` construits à partir des labels (décalage à droite).

    Nécessaire pour NLLB (M2M100) : le collator de transformers ne crée les `decoder_input_ids` que si le modèle expose
    `prepare_decoder_input_ids_from_labels`, ce qui n'est pas le cas de M2M100 (ni d'un modèle enveloppé par PEFT).
    Or avec `label_smoothing_factor > 0`, le Trainer retire `labels` du batch avant l'appel du modèle : celui-ci n'a
    alors plus aucune entrée de décodeur (« You cannot specify both decoder_input_ids and decoder_inputs_embeds »)."""

    def __init__(self, tok, model):
        from transformers import DataCollatorForSeq2Seq
        cfg = (model.get_base_model() if hasattr(model, "get_base_model") else model).config
        self.inner = DataCollatorForSeq2Seq(tok)
        self.pad_id, self.start_id = cfg.pad_token_id, cfg.decoder_start_token_id

    def __call__(self, features):
        b = self.inner(features)
        lab = b["labels"].masked_fill(b["labels"] == -100, self.pad_id)
        dec = lab.new_full(lab.shape, self.pad_id)
        dec[:, 1:] = lab[:, :-1]
        dec[:, 0] = self.start_id
        b["decoder_input_ids"] = dec
        return b


def add_asr_noise(df: pd.DataFrame, asr_hyp_col: str, p: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Amélioration du pipeline : on entraîne la MT en partie sur des sorties ASR bruitées
    (au lieu des seules transcriptions parfaites), pour la rendre robuste aux erreurs de l'étape 1."""
    rng = np.random.default_rng(seed)
    mask = rng.random(len(df)) < p
    out = df.copy()
    out.loc[mask, "transcription"] = out.loc[mask, asr_hyp_col]
    return out


# ==========================================================================
# SeamlessM4T v2 (parole -> texte, zero-shot)
# ==========================================================================
SEAMLESS_TGT = {"eng_Latn": "eng", "fra_Latn": "fra", "arb_Arab": "arb"}


def load_seamless(model_id: str = "facebook/seamless-m4t-v2-large"):
    from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
    proc = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE).eval()
    return proc, model


@torch.inference_mode()
def seamless_s2tt(proc, model, wavs: list[np.ndarray], tgt_lang: str = "eng", batch_size: int = 8) -> list[str]:
    out = []
    for i in range(0, len(wavs), batch_size):
        batch = wavs[i:i + batch_size]
        try:   # l'argument s'appelle `audio` dans les versions récentes de transformers, `audios` avant
            inp = proc(audio=batch, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        except TypeError:
            inp = proc(audios=batch, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
        inp = {k: (v.to(DEVICE, dtype=model.dtype) if v.dtype.is_floating_point else v.to(DEVICE)) for k, v in inp.items()}
        gen = model.generate(**inp, tgt_lang=tgt_lang, num_beams=4, max_new_tokens=256)
        out.extend(x.strip() for x in proc.batch_decode(gen, skip_special_tokens=True))
    return out


# ==========================================================================
# Encodeur de parole + décodeur de texte (diagnostic du papier)
# ==========================================================================
def build_speech_encoder_decoder(encoder_id: str = "facebook/w2v-bert-2.0",
                                 decoder_id: str = "facebook/mbart-large-50", freeze_encoder_layers: int = 12):
    """Le papier rapporte < 5 BLEU en branchant w2v-BERT/XLSR sur le décodeur NLLB, sans explication.
    Hypothèses testées ici :
      (1) désalignement des longueurs : l'encodeur produit ~50 trames/s -> on ajoute l'adaptateur
          convolutif de w2v-BERT (sous-échantillonnage x8) ;
      (2) cross-attention initialisée aléatoirement + LR trop élevé pour l'encodeur ->
          on gèle les premières couches et on utilise un LR plus élevé pour le décodeur ;
      (3) NLLB n'existe pas en version 'CausalLM' dans transformers : on utilise le décodeur
          mBART-50 (même famille, supporté par SpeechEncoderDecoderModel).
    """
    from transformers import (AutoTokenizer, SeamlessM4TFeatureExtractor, SpeechEncoderDecoderModel)
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_id, decoder_id, encoder_add_adapter=True, encoder_layerdrop=0.0, encoder_mask_time_prob=0.05)
    tok = AutoTokenizer.from_pretrained(decoder_id, src_lang="ar_AR", tgt_lang="en_XX")
    fe = SeamlessM4TFeatureExtractor.from_pretrained(encoder_id)
    model.config.decoder_start_token_id = tok.convert_tokens_to_ids("en_XX")
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.forced_bos_token_id = tok.convert_tokens_to_ids("en_XX")
    for i, layer in enumerate(model.encoder.encoder.layers):
        if i < freeze_encoder_layers:
            for p in layer.parameters():
                p.requires_grad = False
    return fe, tok, model.to(DEVICE)


@dataclass
class SpeechTextCollator:
    feature_extractor: Any
    tokenizer: Any
    max_len: int = 200

    def __call__(self, batch):
        feats = self.feature_extractor([b["audio"] for b in batch], sampling_rate=SAMPLING_RATE,
                                       return_tensors="pt", padding=True, return_attention_mask=True)
        lab = self.tokenizer(text_target=[b["text"] for b in batch], truncation=True, max_length=self.max_len,
                             padding=True, return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        return {"inputs": feats["input_features"], "attention_mask": feats["attention_mask"], "labels": labels}


def target_info(name: str) -> dict:
    """Infos d'une langue cible à partir de son nom affiché (« Anglais », « Français »...)."""
    return TARGET_LANGUAGES[name]
