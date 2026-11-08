"""
Configuration centrale du projet TEDxTN.

Toutes les valeurs partagées entre les notebooks et l'interface sont ici :
chemins, identifiants de modèles, langues cibles, hyperparamètres par défaut.
On peut surcharger les chemins avec des variables d'environnement
(utile sur Colab avec Google Drive monté).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Sur Colab, on conseille : os.environ["TEDXTN_WORKDIR"] = "/content/drive/MyDrive/tedxtn"
WORKDIR = Path(os.environ.get("TEDXTN_WORKDIR", PROJECT_ROOT / "workdir"))
DATA_DIR = WORKDIR / "data"                 # CSV téléchargés depuis le Hub
AUDIO_DIR = WORKDIR / "audio_full"          # audio complet des talks (wav 16 kHz)
SEGMENTS_DIR = WORKDIR / "audio_segments"   # segments découpés (wav 16 kHz)
MODELS_DIR = WORKDIR / "models"             # checkpoints fine-tunés
LM_DIR = WORKDIR / "lm"                     # modèles de langage n-gram (KenLM)
RESULTS_DIR = Path(os.environ.get("TEDXTN_RESULTS_DIR", PROJECT_ROOT / "results"))  # métriques, tableaux, figures (légers, versionnés)

for _d in (DATA_DIR, AUDIO_DIR, SEGMENTS_DIR, MODELS_DIR, LM_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
HF_DATASET_ID = "fbougares/TEDxTN"
SAMPLING_RATE = 16_000
SEED = 42

# Noms des fichiers CSV dans le dépôt Hugging Face (format vérifié :
#   <split>.csv    -> audio_filename,start_time,end_time,transcription
#   <split>.en.csv -> audio_filename,start_time,end_time,translation)
SPLIT_FILES = {
    "train": ("train.csv", "train.en.csv"),
    "dev": ("dev.csv", "dev.en.csv"),
    "test": ("test.csv", "test.en.csv"),
}

# Chiffres officiels du papier (Table 5), utilisés pour vérifier les splits.
PAPER_SPLIT_STATS = {
    "train": {"talks": 97, "segments": 15_626},
    "dev": {"talks": 5, "segments": 731},
    "test": {"talks": 6, "segments": 842},
}

# Résultats du papier, affichés comme repères dans les tableaux comparatifs.
PAPER_ASR_RESULTS = {  # Table 6 (test)
    "Whisper-small (zero-shot)": {"WER": 183.81, "CER": 142.00},
    "Whisper-large-v3 (zero-shot)": {"WER": 94.00, "CER": 67.90},
    "Whisper-small (FT)": {"WER": 27.78, "CER": 13.38},
    "Whisper-medium (FT)": {"WER": 25.37, "CER": 13.00},
    "Whisper-large-v3 (FT)": {"WER": 25.19, "CER": 12.33},
    "MMS 1B (FT)": {"WER": 27.91, "CER": 11.27},
    "w2v-BERT-2.0 (FT)": {"WER": 21.37, "CER": 8.34},
}
PAPER_AST_RESULTS = {  # Table 7 (test, BLEU)
    "Whisper-small (zero-shot)": 5.70,
    "Whisper-large-v3 (zero-shot)": 13.95,
    "Whisper-small (FT)": 18.53,
    "Whisper-medium (FT)": 24.50,
    "Whisper-large-v3 (FT)": 25.68,
}

# --------------------------------------------------------------------------
# Langues cibles de la traduction
# --------------------------------------------------------------------------
# Codes NLLB-200 (FLORES-200). NLLB possède un code dédié au tunisien : aeb_Arab.
NLLB_SRC_LANG = "aeb_Arab"
TARGET_LANGUAGES = {
    "Anglais": {"nllb": "eng_Latn", "whisper": "en", "has_reference": True, "llm_name": "English"},
    "Français": {"nllb": "fra_Latn", "whisper": None, "has_reference": False, "llm_name": "French"},
    "Arabe standard (MSA)": {"nllb": "arb_Arab", "whisper": None, "has_reference": False,
                              "llm_name": "Modern Standard Arabic"},
}

# --------------------------------------------------------------------------
# Modèles
# --------------------------------------------------------------------------
BASE_MODELS = {
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
    "w2v-bert-2.0": "facebook/w2v-bert-2.0",
    "mms-1b-all": "facebook/mms-1b-all",
    "nllb-600m": "facebook/nllb-200-distilled-600M",
    "nllb-1.3b": "facebook/nllb-200-distilled-1.3B",
    "seamless-m4t-v2": "facebook/seamless-m4t-v2-large",
    "mbart-50": "facebook/mbart-large-50",
    "llm": "Qwen/Qwen2.5-7B-Instruct",
    "llm-small": "Qwen/Qwen2.5-1.5B-Instruct",
    "comet": "Unbabel/wmt22-comet-da",
}


def env_flag(name: str, default: bool) -> bool:
    """Interrupteur RUN_* d'un notebook, surchargeable par TEDXTN_RUN_<NAME>=0/1 (exécution non interactive)."""
    v = os.environ.get(f"TEDXTN_RUN_{name.upper()}")
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "oui")


def gpu_memory_gb() -> float:
    """VRAM totale du GPU (0 sans GPU). Sert à choisir les modèles et les batchs (plans A/B du README)."""
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    except Exception:
        return 0.0


# Essai rapide de tous les chemins de code (smoke test) : petits sous-ensembles, quelques dizaines de steps.
# Les notebooks n'utilisent alors que des talks de TRAIN (pseudo-dev / pseudo-test = talks de train tenus à part) :
# ni le dev ni le test officiels ne sont touchés. Rediriger TEDXTN_WORKDIR / TEDXTN_RESULTS_DIR vers un dossier
# jetable pour ne rien mélanger aux vrais résultats.
SMOKE = os.environ.get("TEDXTN_SMOKE") == "1"


@dataclass
class AppModels:
    """Modèles chargés par l'interface. Chaque entrée peut être un chemin local
    (checkpoint produit par les notebooks) ou un identifiant du Hub.
    Les valeurs par défaut pointent vers les dossiers créés par les notebooks."""
    asr: str = os.environ.get("TEDXTN_ASR_MODEL", str(MODELS_DIR / "asr_best"))
    asr_fallback: str = os.environ.get("TEDXTN_ASR_FALLBACK", BASE_MODELS["whisper-small"])
    ast_e2e: str = os.environ.get("TEDXTN_AST_MODEL", str(MODELS_DIR / "ast_best"))
    ast_e2e_fallback: str = os.environ.get("TEDXTN_AST_FALLBACK", BASE_MODELS["whisper-small"])
    mt: str = os.environ.get("TEDXTN_MT_MODEL", str(MODELS_DIR / "mt_best"))
    mt_fallback: str = os.environ.get("TEDXTN_MT_FALLBACK", BASE_MODELS["nllb-600m"])
    llm: str = os.environ.get("TEDXTN_LLM_MODEL", BASE_MODELS["llm-small"])
    use_llm: bool = os.environ.get("TEDXTN_USE_LLM", "0") == "1"


@dataclass
class TrainConfig:
    """Hyperparamètres par défaut, pensés pour un GPU Colab T4 (16 Go).
    Le papier entraîne 80 epochs sur HPC ; ici on vise l'early stopping."""
    learning_rate: float = 1e-5
    lora_learning_rate: float = 1e-4
    batch_size: int = 16
    grad_accum: int = 2
    max_epochs: int = 10
    warmup_ratio: float = 0.05
    eval_steps: int = 250                # évaluation (loss + WER/chrF sur 200 segments du dev) et checkpoint
    early_stopping_patience: int = 3
    fp16: bool = True
    gradient_checkpointing: bool = True
    # Optimiseur du Trainer HF. "adamw_bnb_8bit" (bitsandbytes) divise par ~4 la mémoire des états d'Adam :
    # nécessaire pour Whisper-small en complet sur un GPU de 6 Go. Repli sur "adamw_torch" sans CUDA/bitsandbytes.
    optim: str = os.environ.get("TEDXTN_OPTIM", "adamw_torch")
    max_audio_seconds: float = 30.0      # Whisper : fenêtre de 30 s
    min_audio_seconds: float = 0.3
    use_lora: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    augment: bool = True                 # vitesse + bruit (voir common/augment.py)
    spec_augment: bool = True            # masquage temps/fréquence intégré au modèle
    seed: int = SEED
