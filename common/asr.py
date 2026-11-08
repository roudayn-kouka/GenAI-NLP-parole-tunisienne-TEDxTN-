"""
Modèles de reconnaissance (ASR) et de traduction directe (AST end-to-end) :
chargement, collators, entraînement, inférence.

Deux familles, comme dans le papier :
1. Whisper (encodeur-décodeur) : fine-tuning complet ou LoRA, tâche `transcribe` (ASR)
   ou `translate` (audio tunisien -> anglais, AST end-to-end).
2. Encodeurs auto-supervisés + CTC : w2v-BERT-2.0 (meilleur modèle du papier), MMS, XLSR.
   Décodage glouton ou beam search avec modèle de langage n-gram KenLM (amélioration :
   le papier n'utilise aucun LM).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .augment import Augmenter
from .config import SAMPLING_RATE, TrainConfig
from .data import load_audio
from .text import NormConfig, normalize_text

# Normalisation appliquée aux cibles CTC (vocabulaire de caractères plus compact) :
# on garde les distinctions orthographiques (alef, ya...) que la référence utilise.
CTC_NORM = NormConfig(remove_diacritics=True, normalize_alef=False, normalize_ya=False, normalize_gaf=False,
                      arabic_digits=True, collapse_elongation=False, split_mixed_tokens=False,
                      remove_hesitations=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ==========================================================================
# Dataset générique
# ==========================================================================
class SpeechDataset(torch.utils.data.Dataset):
    """Lit les segments audio depuis le disque (colonne `path`) et la cible texte (`text_col`)."""

    def __init__(self, df: pd.DataFrame, text_col: str = "transcription", augmenter: Augmenter | None = None,
                 text_norm: NormConfig | None = None):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.augmenter = augmenter
        self.text_norm = text_norm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = load_audio(row["path"])
        if self.augmenter is not None:
            wav = self.augmenter(wav)
        text = str(row[self.text_col])
        if self.text_norm is not None:
            text = normalize_text(text, self.text_norm)
        return {"audio": wav, "text": text, "seg_id": row.get("seg_id", i)}


# ==========================================================================
# Whisper
# ==========================================================================
def load_whisper(model_id: str, task: str = "transcribe", language: str = "arabic", use_lora: bool = False,
                 load_in_8bit: bool = False, cfg: TrainConfig | None = None, for_training: bool = True):
    """Charge un Whisper (base du Hub ou checkpoint local, éventuellement adaptateur LoRA).

    - SpecAugment activé à l'entraînement (amélioration contre le sur-apprentissage).
    - LoRA (PEFT) pour fine-tuner medium/large-v3 sur un GPU Colab.
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    cfg = cfg or TrainConfig()
    path = Path(model_id)
    adapter = path.exists() and (path / "adapter_config.json").exists()
    base_id = json.loads((path / "adapter_config.json").read_text())["base_model_name_or_path"] if adapter else model_id

    processor = WhisperProcessor.from_pretrained(model_id if (path.exists() and (path / "preprocessor_config.json").exists()) else base_id)
    kwargs = {}
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    model = WhisperForConditionalGeneration.from_pretrained(base_id, **kwargs)

    if for_training and cfg.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10

    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_id, is_trainable=for_training)
    elif use_lora and for_training:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=cfg.lora_target_modules, bias="none")
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    if for_training and cfg.gradient_checkpointing and not use_lora:
        model.config.use_cache = False
    if not load_in_8bit:
        model = model.to(DEVICE)
    return processor, model


@dataclass
class WhisperCollator:
    """Transforme une liste d'exemples en batch : log-Mel (80 ou 128 bandes) + labels tokenisés."""
    processor: Any
    task: str = "transcribe"
    language: str = "arabic"
    max_label_tokens: int = 440

    def __post_init__(self):
        self.processor.tokenizer.set_prefix_tokens(language=self.language, task=self.task)

    def __call__(self, batch):
        feats = self.processor.feature_extractor([b["audio"] for b in batch], sampling_rate=SAMPLING_RATE,
                                                 return_tensors="pt")
        labels = self.processor.tokenizer([b["text"] for b in batch], truncation=True,
                                          max_length=self.max_label_tokens, padding=True, return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (lab[:, 0] == start).all():
            lab = lab[:, 1:]   # le modèle ajoute lui-même le token de début
        return {"input_features": feats["input_features"], "labels": lab}


@torch.inference_mode()
def whisper_generate(model, processor, wavs: list[np.ndarray], task: str = "transcribe", language: str = "arabic",
                     batch_size: int = 16, num_beams: int = 1, max_new_tokens: int = 225,
                     return_details: bool = False):
    """Inférence Whisper par batch. Avec `return_details`, renvoie pour chaque audio :
    texte, tokens, log-probabilités par token et confiance moyenne (utilisé par l'interface)."""
    model.eval()
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    outs = []
    for i in range(0, len(wavs), batch_size):
        chunk = wavs[i:i + batch_size]
        feats = processor.feature_extractor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        x = feats["input_features"].to(dev, dtype=dtype)
        gen = model.generate(input_features=x, task=task, language=language, num_beams=num_beams,
                             max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                             output_scores=return_details)
        texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)
        if not return_details:
            outs.extend(t.strip() for t in texts)
            continue
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        trans = base.compute_transition_scores(gen.sequences, gen.scores,
                                               getattr(gen, "beam_indices", None), normalize_logits=num_beams == 1)
        n_gen = trans.shape[1]
        for seq, sc, txt in zip(gen.sequences, trans, texts):
            gen_ids = seq[-n_gen:]
            toks, probs = [], []
            for tid, s in zip(gen_ids.tolist(), sc.tolist()):
                if tid in processor.tokenizer.all_special_ids or not math.isfinite(s):
                    continue
                toks.append(processor.tokenizer.decode([tid]))
                probs.append(float(np.exp(s)))
            outs.append({"text": txt.strip(), "tokens": toks, "token_probs": probs,
                         "confidence": float(np.mean(probs)) if probs else 0.0})
    return outs


# ==========================================================================
# CTC : w2v-BERT-2.0 / MMS / XLSR
# ==========================================================================
def build_ctc_vocab(texts: list[str], out_dir: str | Path) -> dict:
    """Vocabulaire de caractères appris sur le train (arabe + latin + chiffres)."""
    chars = set()
    for t in texts:
        chars.update(normalize_text(t, CTC_NORM))
    chars.discard(" ")
    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    return vocab


def load_ctc(backbone: str, vocab_dir: str | Path, cfg: TrainConfig | None = None, from_checkpoint: bool = False):
    """Charge un modèle CTC.
    backbone : 'facebook/w2v-bert-2.0' (recommandé), 'facebook/mms-1b-all', 'facebook/wav2vec2-xls-r-300m'
    ou un checkpoint local déjà fine-tuné (`from_checkpoint=True`)."""
    from transformers import Wav2Vec2CTCTokenizer
    cfg = cfg or TrainConfig()
    is_bert = "w2v-bert" in backbone or (from_checkpoint and _is_w2vbert_checkpoint(backbone))
    if from_checkpoint:
        if is_bert:
            from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
            proc = Wav2Vec2BertProcessor.from_pretrained(backbone)
            model = Wav2Vec2BertForCTC.from_pretrained(backbone)
        else:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            proc = Wav2Vec2Processor.from_pretrained(backbone)
            model = Wav2Vec2ForCTC.from_pretrained(backbone)
        return proc, model.to(DEVICE)

    tok = Wav2Vec2CTCTokenizer(str(Path(vocab_dir) / "vocab.json"), unk_token="[UNK]", pad_token="[PAD]",
                               word_delimiter_token="|")
    common = dict(attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
                  mask_time_prob=0.05 if cfg.spec_augment else 0.0, ctc_loss_reduction="mean",
                  ctc_zero_infinity=True, pad_token_id=tok.pad_token_id, vocab_size=len(tok))
    if is_bert:
        from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor
        fe = SeamlessM4TFeatureExtractor.from_pretrained(backbone)
        proc = Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tok)
        model = Wav2Vec2BertForCTC.from_pretrained(backbone, add_adapter=True, **common)
    else:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=SAMPLING_RATE, padding_value=0.0,
                                      do_normalize=True, return_attention_mask=True)
        proc = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)
        model = Wav2Vec2ForCTC.from_pretrained(backbone, ignore_mismatched_sizes=True, **common)
        if "mms" in backbone:
            # MMS : on n'entraîne qu'un adaptateur de langue (~2.5 M paramètres), le reste est gelé
            model.init_adapter_layers()
            model.freeze_base_model()
            for name, p in model.named_parameters():
                if "adapter" in name or "lm_head" in name:
                    p.requires_grad = True
        else:
            model.freeze_feature_encoder()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return proc, model.to(DEVICE)


def _is_w2vbert_checkpoint(path) -> bool:
    p = Path(path) / "config.json"
    return p.exists() and "wav2vec2-bert" in json.loads(p.read_text()).get("model_type", "")


@dataclass
class CTCCollator:
    processor: Any

    def __call__(self, batch):
        is_bert = "SeamlessM4T" in type(self.processor.feature_extractor).__name__
        feats = self.processor.feature_extractor([b["audio"] for b in batch], sampling_rate=SAMPLING_RATE,
                                                 return_tensors="pt", padding=True, return_attention_mask=True)
        key = "input_features" if is_bert else "input_values"
        labels = self.processor.tokenizer([normalize_text(b["text"], CTC_NORM) for b in batch],
                                          padding=True, return_tensors="pt")
        lab = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        return {key: feats[key], "attention_mask": feats["attention_mask"], "labels": lab}


def build_kenlm_decoder(processor, kenlm_path: str | Path | None, alpha: float = 0.5, beta: float = 1.0):
    """Décodeur beam search pyctcdecode, avec ou sans LM KenLM (amélioration du papier)."""
    from pyctcdecode import build_ctcdecoder
    vocab = processor.tokenizer.get_vocab()
    labels = [None] * len(vocab)
    for tok, idx in vocab.items():
        labels[idx] = tok
    pad = processor.tokenizer.pad_token
    labels = ["" if t == pad else " " if t == "|" else "⁇" if t == "[UNK]" else t for t in labels]
    return build_ctcdecoder(labels, kenlm_model_path=str(kenlm_path) if kenlm_path else None, alpha=alpha, beta=beta)


@torch.inference_mode()
def ctc_transcribe(model, processor, wavs: list[np.ndarray], decoder=None, batch_size: int = 8,
                   beam_width: int = 64, return_details: bool = False):
    """Inférence CTC. Avec `return_details`, renvoie aussi la séquence d'émissions par trame
    (caractère le plus probable et sa probabilité), affichée par l'interface."""
    model.eval()
    is_bert = "SeamlessM4T" in type(processor.feature_extractor).__name__
    key = "input_features" if is_bert else "input_values"
    outs = []
    for i in range(0, len(wavs), batch_size):
        chunk = wavs[i:i + batch_size]
        feats = processor.feature_extractor(chunk, sampling_rate=SAMPLING_RATE, return_tensors="pt",
                                            padding=True, return_attention_mask=True)
        x = feats[key].to(model.device)
        logits = model(**{key: x, "attention_mask": feats["attention_mask"].to(model.device)}).logits.float()
        logp = torch.log_softmax(logits, -1).cpu().numpy()
        # longueur de sortie réelle de chaque exemple (ratio trames sortie / entrée)
        in_lens = feats["attention_mask"].sum(-1).numpy()
        ratio = logp.shape[1] / feats["attention_mask"].shape[1]
        for lp, L in zip(logp, in_lens):
            lp = lp[: max(1, int(math.ceil(L * ratio)))]
            if decoder is not None:
                text = decoder.decode(lp, beam_width=beam_width)
            else:
                ids = lp.argmax(-1)
                text = processor.decode(ids)
            text = " ".join(text.split())
            if return_details:
                ids = lp.argmax(-1)
                inv = {v: k for k, v in processor.tokenizer.get_vocab().items()}
                frames = [(inv.get(int(t), "?"), float(np.exp(lp[j, t]))) for j, t in enumerate(ids)]
                outs.append({"text": text, "frames": frames,
                             "confidence": float(np.mean([p for c, p in frames if c != processor.tokenizer.pad_token] or [0]))})
            else:
                outs.append(text)
    return outs


# ==========================================================================
# Entraînement
# ==========================================================================
class GenerationWERCallback:
    """Callback : calcule le WER sur un sous-ensemble du dev à chaque évaluation
    (courbe d'apprentissage réelle, en plus de la loss)."""

    def __init__(self, predict_fn, dev_wavs, dev_refs, every_eval: int = 1):
        self.predict_fn, self.dev_wavs, self.dev_refs = predict_fn, dev_wavs, dev_refs
        self.history = []
        self.every_eval, self._n = every_eval, 0

    def as_callback(self):
        from transformers import TrainerCallback
        outer = self

        class _CB(TrainerCallback):
            def on_evaluate(self, args, state, control, model=None, **kw):
                outer._n += 1
                if outer._n % outer.every_eval:
                    return
                from .metrics import wer_cer
                hyps = outer.predict_fn(model, outer.dev_wavs)
                m = wer_cer(outer.dev_refs, hyps)
                outer.history.append({"step": state.global_step, "dev_WER_subset": m["WER"], "dev_CER_subset": m["CER"]})
                print(f"[step {state.global_step}] WER dev (sous-ensemble) = {m['WER']:.2f}  CER = {m['CER']:.2f}")
                model.train()
        return _CB()


def _resolve_optim(name: str) -> str:
    """Les optimiseurs bitsandbytes (*bnb*) exigent CUDA et le paquet bitsandbytes : sinon repli sur adamw_torch."""
    if "bnb" in name or "8bit" in name:
        try:
            import bitsandbytes  # noqa: F401
            if not torch.cuda.is_available():
                raise ImportError("pas de CUDA")
        except Exception as e:  # ImportError ou erreur d'initialisation de bitsandbytes
            print(f"[training_args] optim={name} indisponible ({e}) -> adamw_torch")
            return "adamw_torch"
    return name


def training_args(output_dir: str | Path, cfg: TrainConfig, seq2seq: bool = True, lr: float | None = None,
                  n_train: int | None = None, **overrides):
    """Arguments d'entraînement adaptés à un GPU Colab (fp16, accumulation, early stopping)."""
    from transformers import Seq2SeqTrainingArguments, TrainingArguments
    Args = Seq2SeqTrainingArguments if seq2seq else TrainingArguments
    kw = dict(
        output_dir=str(output_dir), per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size, gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=lr or (cfg.lora_learning_rate if cfg.use_lora else cfg.learning_rate),
        warmup_ratio=cfg.warmup_ratio, num_train_epochs=cfg.max_epochs, fp16=cfg.fp16 and torch.cuda.is_available(),
        eval_steps=cfg.eval_steps, save_steps=cfg.eval_steps, logging_steps=50, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", seed=cfg.seed, dataloader_num_workers=2, remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing and not cfg.use_lora,
        label_names=["labels"], optim=_resolve_optim(cfg.optim),
    )
    import inspect
    params = inspect.signature(Args.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    kw["save_strategy"] = "steps"
    kw.update(overrides)
    return Args(**kw)


def make_trainer(model, args, train_ds, eval_ds, collator, processor, cfg: TrainConfig, extra_callbacks=()):
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Trainer
    import inspect
    T = Seq2SeqTrainer if "Seq2Seq" in type(args).__name__ else Trainer
    kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator,
              callbacks=[EarlyStoppingCallback(cfg.early_stopping_patience), *extra_callbacks])
    params = inspect.signature(T.__init__).parameters
    kw["processing_class" if "processing_class" in params else "tokenizer"] = processor.feature_extractor
    return T(**kw)


def save_for_inference(model, processor, out_dir: str | Path, merge_lora: bool = True):
    """Sauvegarde le modèle final (fusionne LoRA pour un chargement simple dans l'interface)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if merge_lora and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t0
