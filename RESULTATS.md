# Résultats : tedxtn-tunisian-speech

> Document en cours de rédaction. Les tableaux finaux (ASR, traduction, IC 95 %, comparaison au papier) seront
> ajoutés quand tous les systèmes seront évalués. Chaque chiffre vient d'une exécution réelle ; les choix se font sur
> le dev, le test ne sert qu'à l'évaluation finale.

## Notes de protocole

### Whisper-small fine-tuné (ASR) : choix du checkpoint

- Règle fixée **avant** l'entraînement : early stopping et choix du meilleur checkpoint sur la **dev loss**
  (évaluation tous les 250 steps, patience 3).
- L'entraînement s'est arrêté au step 3500 / 4900 (époque 7,1) : la dev loss n'a plus baissé après le step 2750
  (0,4705 ; puis 0,4772, 0,4715, 0,4712). **Modèle retenu : checkpoint-2750.**
- Le checkpoint-3500 avait un WER plus bas sur le sous-ensemble de suivi de 200 segments du dev (38,63 % contre 40,26 %
  pour le checkpoint-2750). Écart non testé et vraisemblablement dans le bruit de ce sous-ensemble. **La règle a été
  respectée** : le checkpoint-3500 n'a pas été évalué pour choisir, et il a été supprimé.
- Courbe de suivi : les WER des steps 250 à 750 ont été perdus lors d'un arrêt volontaire de l'entraînement (bug
  corrigé, CHANGELOG_EXECUTION.md n° 46) ; ceux des steps 1000 et 1250 ont été recalculés depuis les checkpoints.

### E2E-2 (Whisper-small en traduction, initialisé depuis l'ASR fine-tuné) : choix du checkpoint

- Même règle que pour l'ASR, fixée avant l'entraînement : early stopping et meilleur checkpoint sur la **dev loss**
  (évaluation tous les 250 steps, patience 3).
- Arrêt au step 2000 / 3920 : la dev loss n'a plus baissé après le step 1250. **Modèle retenu : checkpoint-1250.**

  | step | dev loss | BLEU (200 seg. du dev) | chrF++ (200 seg. du dev) |
  |---|---|---|---|
  | 250 | 1,9119 | 14,67 | 38,31 |
  | 500 | 1,7993 | 15,64 | 40,54 |
  | 750 | 1,7659 | 18,33 | 41,96 |
  | 1000 | 1,7417 | 18,50 | 42,26 |
  | **1250** | **1,7358** | 16,85 | 41,35 |
  | 1500 | 1,7547 | 18,26 | 42,61 |
  | 1750 | 1,7489 | 17,56 | 42,25 |
  | 2000 | 1,7748 | 17,57 | 42,51 |

- **Limite** : en traduction, la dev loss remonte après le step 1250 (le modèle devient trop sûr de lui : la
  cross-entropie pénalise fortement ses erreurs commises avec confiance) alors que le BLEU et le chrF++ restent
  stables (16,9-18,5 et 41,4-42,6 sur 200 segments). Le choix sur la loss est donc moins adapté qu'en ASR : le
  checkpoint retenu a le BLEU de suivi le plus bas des steps 750 à 2000. **La règle a été gardée** ; c'est une limite
  du protocole, pas une raison de changer de checkpoint après coup. Une règle sur une métrique de génération (BLEU /
  chrF++ sur tout le dev) serait à fixer avant l'entraînement dans une prochaine expérience.
- Modèle retenu sur **tout le dev** (730 segments : le segment `dev__00572`, sans traduction de référence, est
  exclu de l'évaluation de la traduction ; beam 4) : **BLEU 19,08** [17,53 ; 20,60] (protocole du papier : casse
  gardée, ponctuation retirée ; sacreBLEU 13a standard : 18,80), **chrF++ 43,61** [42,39 ; 44,90] (IC 95 % par
  bootstrap sur les segments, 1 000 tirages).

### Normalisation d'évaluation (décidée sur le dev, avant toute évaluation sur le test)

- Trois mesures : **WER brut** (`RAW_NORM`, protocole du papier : ponctuation et casse latine ignorées), **WER
  normalisé** (`EVAL_NORM` : orthographe seule, c.-à-d. alef, ya, diacritiques, /g/, chiffres, allongements) et
  **WER sans hésitations** (`EVAL_NORM_NOHES` : `EVAL_NORM` + hésitations retirées de la référence et de
  l'hypothèse, repérées sur la forme d'origine du mot ; « أم » est un mot et reste).
- Retirées de l'évaluation après une ablation règle par règle sur le dev (`scripts/norm_ablation.py`,
  `results/norm_ablation_dev.csv`) : la frontière de script (`الfumoir → ال fumoir`) est asymétrique (les références
  écrivent les emprunts en arabe ; seules les hypothèses ont des mots hybrides, coupés en deux erreurs) ; elle reste
  une convention d'écriture (notebook 03). Le retrait des hésitations fait **monter** le WER (whisper-small-ft, dev :
  -82 erreurs mais -433 mots de référence) : les hésitations sont surtout bien reconnues, c'est un effet de
  dénominateur ; d'où une colonne à part.
- **Constat (dev, whisper-small-ft)** : la normalisation orthographique seule ne fait gagner que **0,26 point**
  (WER brut 34,80 % → normalisé 34,54 % ; 29 erreurs de moins sur 3 970, toutes dues à l'alef). Les erreurs du
  modèle portent donc sur le fond (mots, emprunts, segmentation) bien plus que sur l'orthographe. **À confirmer sur
  le test.** (Le chiffre de 34,57 % cité avant la décision venait d'une configuration intermédiaire : ancienne
  `EVAL_NORM` sans les hésitations mais avec la frontière de script.)

### Écarts au protocole du papier (à compléter)

- Train : 15 703 segments dans `train.csv` du Hub contre 15 626 dans le papier (+77), gardés.
- NLLB-600M fine-tuné en LoRA (r = 16 sur q/k/v/out_proj et fc1/fc2, poids fp32, précision mixte fp16) au lieu d'un
  fine-tuning complet (GPU de 6 Go).
