# Parole tunisienne : reconnaissance et traduction (TEDxTN)

Projet commun **GenAI / NLP** : reconnaissance automatique de la parole tunisienne (ASR) et traduction de la
parole (vers l'anglais, le français et l'arabe standard) sur le corpus **TEDxTN**, avec les deux extensions
(normalisation, identification dialectale) et une interface web qui montre chaque étape du traitement.

- Corpus : [fbougares/TEDxTN](https://huggingface.co/datasets/fbougares/TEDxTN) (CC BY-NC-ND 4.0)
- Article de référence : Bougares et al., *TEDxTN: A Three-way Speech Translation Corpus for Code-Switched
  Tunisian Arabic - English*, ArabicNLP 2025 ([PDF](https://aclanthology.org/2025.arabicnlp-main.22.pdf))

---

## 1. Organisation du dépôt

```
tedxtn-tunisian-speech/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_Tunisian_ASR.ipynb               # obligatoire : ASR
│   ├── 02_Speech_to_Text_Translation.ipynb # obligatoire : traduction directe vs pipeline
│   ├── 03_Text_Normalization.ipynb         # extension 1
│   ├── 04_Dialect_Identification.ipynb     # extension 2
│   └── 05_Interface_Demo.ipynb             # lance l'interface depuis un notebook
├── common/                                 # code partagé par les notebooks et l'interface
│   ├── config.py      chemins, modèles, langues, hyperparamètres
│   ├── data.py        annotations, audio YouTube, découpage, contrôle des fuites, prédictions
│   ├── text.py        script, code-switching, CMI, normalisation, comparaison inter-scripts
│   ├── metrics.py     WER/CER bruts et normalisés, BLEU, chrF++, COMET, bootstrap, analyse d'erreurs
│   ├── augment.py     augmentation audio (vitesse, bruit, gain)
│   ├── asr.py         Whisper (complet / LoRA), w2v-BERT-2.0 / MMS (CTC), KenLM
│   ├── translate.py   NLLB, SeamlessM4T, encodeur parole + décodeur texte
│   ├── llm.py         post-correction, traduction, normalisation par LLM (GenAI)
│   └── pipeline.py    pipeline complet avec trace étape par étape
├── app/app.py                              # interface Gradio « Derja Voice Lab »
├── scripts/
│   ├── prepare_data.py                     # annotations + audio + segments en une commande
│   └── build_kenlm.sh                      # modèle de langage 4-gram (train uniquement)
├── tests/test_core.py                      # tests sans GPU (format réel des données)
└── results/                                # tableaux CSV, figures, prédictions (results/preds/)
```

Les fichiers lourds ne sont **pas** dans le dépôt : audio, segments et modèles vont dans `workdir/`
(ou `MyDrive/tedxtn` sur Colab, via la variable `TEDXTN_WORKDIR`).

## 2. Démarrage rapide

### Sur Google Colab (recommandé)
1. Copier ce dossier dans `MyDrive/tedxtn-tunisian-speech` (ou le pousser sur GitHub et adapter `REPO_URL`
   dans la première cellule de chaque notebook).
2. Ouvrir un notebook, choisir un GPU (*Exécution > Modifier le type d'exécution*), tout exécuter.
   La première cellule monte Drive, installe les dépendances et règle `TEDXTN_WORKDIR=/content/drive/MyDrive/tedxtn`.

### En local
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + ffmpeg installé sur le système
python scripts/prepare_data.py           # télécharge annotations et audio, découpe les segments
jupyter lab notebooks/
python app/app.py                        # interface sur http://localhost:7860
```

## 3. Ordre d'exécution

| Ordre | Notebook | Produit | Utilisé par |
|---|---|---|---|
| 1 | `01_Tunisian_ASR` | segments audio, `models/asr_best`, `results/asr_results.csv`, prédictions ASR | 02, 03, 05, interface |
| 2 | `02_Speech_to_Text_Translation` | `models/ast_best`, `models/mt_best`, `results/ast_results.csv` | 05, interface |
| 3 | `03_Text_Normalization` | `results/normalization_results.csv`, fiches d'annotation | — |
| 4 | `04_Dialect_Identification` | `results/dialect_id_results.csv` | — |
| 5 | `05_Interface_Demo` | lien public vers l'interface | démonstration |

Les notebooks 03 et 04 peuvent tourner en parallèle de 01-02 (03 n'a besoin que des annotations ; il
réutilise les prédictions ASR si elles existent).

**Reprise et travail en parallèle.** Chaque système enregistre ses sorties dans
`results/preds/<tâche>__<système>__<split>.csv`. Les sections « Résultats » rassemblent tous les fichiers
présents : chaque membre de l'équipe peut entraîner un modèle sur son propre compte Colab et déposer ses
prédictions, puis n'importe qui relance l'évaluation. Les entraînements reprennent automatiquement depuis le
dernier checkpoint après une coupure de Colab.

Interrupteurs en tête de chaque notebook (`RUN_WHISPER_FT`, `RUN_CTC_FT`, `RUN_LLM`, ...) : ils sont
désactivés automatiquement sans GPU, ce qui permet de rejouer l'exploration et l'évaluation sur CPU.

## 4. Répartition proposée (4 personnes)

| Personne | Travail | Fichiers |
|---|---|---|
| 1 | Données, exploration, baselines zero-shot, README | `01` sections 3-5, `common/data.py`, `scripts/` |
| 2 | Fine-tuning ASR (Whisper, w2v-BERT + KenLM), analyse d'erreurs, LLM | `01` sections 6-11, `common/asr.py` |
| 3 | Traduction : direct vs pipeline, NLLB, LLM, COMET | `02`, `common/translate.py`, `common/llm.py` |
| 4 | Normalisation, identification dialectale, interface, intégration | `03`, `04`, `05`, `app/`, `common/pipeline.py` |

Un notebook = un propriétaire (les `.ipynb` fusionnent mal dans git).

## 5. Récupération de l'audio

La licence interdit de redistribuer l'audio : le papier fournit l'URL YouTube de chaque talk.
`common/data.py` cherche ces URL dans le dépôt Hugging Face, puis par titre sur YouTube en dernier recours, et
écrit `workdir/data/youtube_urls.csv`. **Ce fichier est la source de vérité** : les URL marquées
`ytsearch (À VÉRIFIER)` doivent être contrôlées à la main. Un rapport (`audio_report.csv`) signale tout audio
plus court que la dernière annotation (mauvaise vidéo probable).

Si YouTube bloque Colab (« confirm you're not a bot ») : exporter les cookies du navigateur au format Netscape
et définir `YTDLP_COOKIES=/chemin/cookies.txt`, ou lancer `scripts/prepare_data.py` en local puis copier
`workdir/audio_full/` sur Drive.

## 6. Améliorations implémentées (par rapport au papier)

| Limite | Amélioration | Où |
|---|---|---|
| Peu de données (~22 h) | augmentation vitesse/bruit/gain + SpecAugment | `common/augment.py`, `asr.py` |
| Calcul limité | LoRA (8 bits pour large-v3), fp16, gradient checkpointing, early stopping, reprise | `asr.py`, 01, 02 |
| Pas de modèle de langage | KenLM 4-gram + beam search, α/β réglés sur le dev | `scripts/build_kenlm.sh`, 01 §7 |
| Orthographe non standard | WER normalisé + WER script-aware (emprunts dans l'autre script) | `text.py`, `metrics.py` |
| Code-switching | WER/chrF par type de segment TUN/MIXED/FOR, CMI, mots déclencheurs | 01 §3.4 et §9.1, 02 §9.2 |
| Dev/test petits | IC à 95 % par bootstrap, test apparié direct vs pipeline | `metrics.py`, 01 §8, 02 §8.1 |
| Pas de comparaison direct/pipeline | 2 directs + 5 pipelines (dont oracle) sur le même test | 02 |
| MT sensible aux erreurs ASR | NLLB fine-tuné avec sorties ASR bruitées | 02 §7.1 |
| BLEU seul | chrF++, COMET-22, analyse manuelle guidée | 02 §8-9 |
| Échec encodeur + NLLB inexpliqué | diagnostic : adaptateur, gel partiel, décodeur mBART | 02 §7.3 |
| Volet GenAI | LLM pour post-correction ASR, traduction, normalisation | `llm.py`, 01 §10, 02 §7.2, 03 §5.4 |
| Langues cibles | français et arabe standard (NLLB, LLM, pivot), évaluation aller-retour | 02 §10, interface |

## 7. L'interface « Derja Voice Lab »

```bash
python app/app.py                 # local
python app/app.py --share         # lien public
python app/app.py --asr workdir/models/asr_best --mt facebook/nllb-200-distilled-600M
```
ou depuis le notebook `05_Interface_Demo.ipynb`. On enregistre sa voix (ou on charge un fichier), on choisit la
langue cible (anglais, français, arabe standard) et la méthode (pipeline, directe ou comparaison). L'interface
affiche, au fur et à mesure, les 9 étapes : signal d'entrée, détection de parole, spectrogramme log-Mel,
reconnaissance (confiance par token), code-switching, normalisation règle par règle, post-correction LLM
(option), traduction(s) et temps par étape. L'onglet « Résultats des expériences » affiche les tableaux
produits par les notebooks.

Si un modèle fine-tuné est absent, l'interface utilise le modèle de base et l'indique clairement dans l'étape
concernée. Variables utiles : `TEDXTN_ASR_MODEL`, `TEDXTN_AST_MODEL`, `TEDXTN_MT_MODEL`, `TEDXTN_LLM_MODEL`.

## 8. Reproductibilité

- Graine fixe (`SEED = 42`) pour les tirages, l'augmentation et l'entraînement.
- Splits officiels du papier, vérifiés (effectifs, aucune fuite de talk, de texte ou de segment).
- Toute décision (checkpoint, paramètres du LM, meilleur système) est prise sur le dev ; le test sert une fois.
- Versions : voir `requirements.txt` ; `transformers` est borné à `<5` (changements d'API).
- Tests sans GPU : `python tests/test_core.py`.

## 9. Licence et éthique

Les annotations et l'audio de TEDxTN sont sous **CC BY-NC-ND 4.0** : usage non commercial, pas de
redistribution de l'audio ni de versions modifiées. Par prudence, garder les modèles fine-tunés et les
prédictions dans un espace privé (Drive, dépôt GitHub ou Hub privé) et ne pas publier l'audio téléchargé.

## 10. Références principales

Bougares et al. (2025) TEDxTN · Radford et al. (2022) Whisper · Seamless Communication (2023) w2v-BERT 2.0 et
SeamlessM4T · Pratap et al. (2023) MMS · NLLB Team (2022) · Hu et al. (2021) LoRA · Park et al. (2019)
SpecAugment · Heafield (2011) KenLM · Post (2018) sacreBLEU · Popović (2017) chrF++ · Rei et al. (2022) COMET ·
Zribi et al. (2014) CODA-TUN · Xue et al. (2022) ByT5 · Qwen Team (2024) Qwen2.5.
