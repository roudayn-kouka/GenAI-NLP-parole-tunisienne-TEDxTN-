# Journal d'exécution réelle

Corrections et adaptations faites lors du premier passage sur données réelles, GPU et vraies bibliothèques.
Chaque entrée correspond à un commit git.

## Environnement

- Machine : Windows 11, i7-13650HX, 32 Go RAM, NVIDIA RTX 3050 Laptop 6 Go (pilote 581.86, CUDA 13.0).
- Exécution sous WSL2 Ubuntu 22.04, Python 3.10.12, venv `~/tedxtn-venv`, workdir `~/tedxtn-workdir`.
- Plan B : Whisper-small et le reste en local ; w2v-BERT-2.0 (fine-tuning complet, ~600 M paramètres,
  ne tient pas dans 6 Go) sur Google Colab. Whisper-large-v3 LoRA et Qwen2.5-7B sautés (VRAM insuffisante).

## Corrections et adaptations

| # | Fichier | Problème | Correction |
|---|---|---|---|
| 1 | `.gitattributes` | Git sous Windows convertit en CRLF ; `scripts/build_kenlm.sh` lancé sous WSL échouerait. | Fins de ligne LF forcées. |
| 2 | `common/config.py`, `common/asr.py` | Optimiseur non configurable (AdamW fp32 par défaut) : les états d'Adam de Whisper-small (~2 Go) ne laissent pas assez de place sur 6 Go. | Champ `TrainConfig.optim` (défaut `adamw_torch`, surcharge `TEDXTN_OPTIM`) transmis à `training_args` ; repli automatique sur `adamw_torch` si bitsandbytes ou CUDA absent. |
