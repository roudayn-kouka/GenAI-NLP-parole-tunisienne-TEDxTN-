# Résultats : tedxtn-tunisian-speech

> Chaque chiffre vient d'une exécution réelle (notebooks exécutés dans `/root/runs/*_final.ipynb`, prédictions dans
> `results/preds/`, tableaux dans `results/*.csv`). Tous les choix (checkpoints, règles, meilleur système) ont été faits
> sur le **dev** ; le **test** a été évalué une seule fois, lors de la passe finale du 12/11/2026, et aucun système n'a
> été modifié ensuite. IC 95 % : bootstrap sur les segments (1 000 tirages). Reste à faire : section 6
> (normalisation, ByT5), voir « À faire ».

## 1. Conditions d'exécution

- Machine : Windows 11, i7-13650HX, 32 Go de RAM, NVIDIA RTX 3050 Laptop **6 Go** ; WSL2 Ubuntu 22.04, Python 3.10,
  torch 2.6, transformers 4.57. Tout a tourné en local (aucun calcul sur Colab).
- Contraintes qui expliquent les écarts au papier : pas de modèle > ~1 B paramètres en fine-tuning, NLLB-600M en
  LoRA, AdamW 8 bits, batch par GPU plafonné à 4 (accumulation pour garder le batch effectif), LLM Qwen2.5-1.5B
  quantifié en 4 bits (le 7B prévu ne tient pas), pas de w2v-BERT ni d'E2E-1 (voir « Améliorations envisagées »).
- Entraînements (un seul essai par système, graine 42 ; early stopping et checkpoint choisis sur la dev loss,
  évaluation tous les 250 steps pour Whisper, 400 pour NLLB, patience 3) :

  | Système | Initialisation | Arrêt | Checkpoint retenu |
  |---|---|---|---|
  | ASR whisper-small-ft | openai/whisper-small | step 3500 / 4900 (epoch 7,1), early stopping | 2750 (≈ 5,6 epochs) |
  | E2E-2 (traduction directe) | ASR whisper-small-ft | step 2000 / 3920 (epoch 4,1), early stopping | 1250 (≈ 2,6 epochs) |
  | C2 : NLLB-600M LoRA | facebook/nllb-200-distilled-600M | 6 epochs complètes (2940 steps) | 2800 |
  | C3 : NLLB-600M LoRA, 50 % de sources ASR bruitées | idem | 6 epochs complètes (2940 steps) | 2800 |

- Décodage : Whisper ASR greedy (beam 5 en variante, test seulement), traduction beam 4.
- Licence CC BY-NC-ND : ni l'audio ni les modèles ne sont publiés ; le dépôt reste privé.

## 2. Données

| split | talks | segments | heures | segments papier | heures papier | segments traduits (traduction) |
|---|---|---|---|---|---|---|
| train | 97 | 15 703 (15 676 après nettoyage) | 22,67 | 15 626 | 22 h 40 | 15 666 |
| dev | 5 | 731 | 1,13 | 731 | 1 h 07 | 730 |
| test | 6 | 842 | 1,24 | 842 | 1 h 14 | 841 |

- Aucun talk ni transcription (≥ 6 mots) partagé entre splits ; une paire de segments se chevauche dans le temps ;
  splits officiels non modifiés. 27 segments du train écartés de
  l'entraînement (22 < 0,3 s, 4 > 30 s, 1 vide). Un segment du dev et un du test n'ont pas de traduction de
  référence : exclus de l'évaluation de la traduction seulement.
- Segments avec au moins un mot étranger : 53,8 % (papier : 53,7 %). L'indice CMI calculé ici (train 11,1 ; dev 16,7 ;
  test 15,8) est plus bas que celui du papier (20,66 ; 24,37 ; 33,09) : la définition exacte du papier n'est pas
  donnée, les deux ne sont pas comparables.

## 3. Reconnaissance de la parole (ASR)

### 3.1 Test (842 segments)

| Système | WER brut [IC 95 %] | CER | WER normalisé | WER sans hésitations | WER script-aware | sub / del / ins (%) |
|---|---|---|---|---|---|---|
| **whisper-small-ft** (retenu sur le dev) | **34,55** [32,4 ; 37,0] | **16,91** | 34,31 | 33,72 | 33,66 | 26,1 / 3,6 / 4,8 |
| whisper-small-ft, beam 5 (variante) | 32,06 [30,1 ; 34,4] | 15,75 | 31,80 | 32,00 | 31,15 | 24,4 / 3,7 / 4,1 |
| whisper-small-ft + post-correction LLM | 42,70 [40,7 ; 45,3] | 22,74 | 42,15 | 41,51 | 41,28 | 31,7 / 5,8 / 5,3 |
| Whisper-large-v3 zero-shot | 99,74 [95,3 ; 104,9] | 76,71 | 99,28 | 99,47 | 98,45 | 58,3 / 30,0 / 11,4 |
| Whisper-small zero-shot | 117,61 [111,4 ; 124,7] | 89,60 | 117,21 | 118,03 | 116,29 | 63,4 / 26,7 / 27,5 |

- Dev (731 segments, whisper-small-ft) : WER brut 34,80 [31,98 ; 39,50], CER 15,94, normalisé 34,54, sans
  hésitations 35,16. Dev et test sont cohérents.
- **Beam 5** : 2,5 points de WER de moins sur le test, mais cette variante n'a été décodée que sur le test ; elle
  ne pouvait donc pas être choisie (règle : choix sur le dev). Le système retenu reste le décodage greedy.
- **Normalisation** : sur le test aussi, l'orthographe seule ne fait gagner que 0,24 point (34,55 → 34,31 ; dev :
  0,26) : constat du dev **confirmé**. Retirer les hésitations baisse le WER du test (33,72) alors qu'il le montait
  sur le dev (35,16) : l'effet de dénominateur dépend de la part d'hésitations bien reconnues, il n'est pas stable.
- **Post-correction LLM** (non retenue sur le dev) : sur le test, +8,17 points de WER, IC 95 % apparié
  [+7,19 ; +9,12], 71,4 % des segments modifiés : la dégradation observée sur le dev est **confirmée**.
- Gain du fine-tuning : 65,2 points de WER par rapport au meilleur zero-shot (Whisper-large-v3).

### 3.2 Par type de segment (test, whisper-small-ft)

| Type | segments | WER | CER | WER sans hésitations | papier (w2v-BERT-2.0, Tableau 9) |
|---|---|---|---|---|---|
| TUN (tunisien seul) | 215 | 45,29 | 20,83 | 40,38 | 24,16 |
| MIXED (code-switching) | 538 | 34,18 | 17,02 | 34,21 | 21,31 |
| FOR (langue étrangère seule) | 89 | 15,27 | 7,14 | 14,40 | 13,93 |
| Tous | 842 | 34,55 | 16,91 | 33,72 | 21,37 |

- Les segments en tunisien seul sont les plus difficiles, comme dans le papier ; l'écart avec le papier est surtout
  sur le tunisien (+21 points) et peu sur le français / anglais seul (+1,3 point). Le classement des segments en
  TUN / MIXED / FOR est recalculé ici (215 / 538 / 89 contre 215 / 551 / 76 dans le papier) ; le papier ne donne ce
  tableau que pour w2v-BERT.
- Erreurs (test, catégories automatiques) : proches phonétiquement 25,9 %, lexicales 24,7 %, emprunts /
  code-switching 22,5 %, insertions arabes 9,4 %, suppressions arabes 7,2 %, insertions latines 4,5 %, suppressions
  latines 3,3 %, même mot dans l'autre écriture 1,9 %, variantes orthographiques 0,5 %.
- Talks difficiles (notebook 01 §9.5) : le WER par talk est surtout corrélé au bruit (Spearman WER / SNR estimé :
  -0,63 sur 50 talks ; débit, CMI : ≈ 0). Les 6 talks TEDx Pasha Street du train ont un WER moyen de 100 % (SNR
  4,4 dB) contre 24,6 % pour les autres talks échantillonnés (SNR 25,5 dB) : enregistrement très bruité, gardé dans
  le train (splits non modifiés).

### 3.3 Comparaison avec le papier (Tableau 6)

| Système | dev WER / CER (ici) | dev WER / CER (papier) | test WER / CER (ici) | test WER / CER (papier) |
|---|---|---|---|---|
| Whisper-small zero-shot | — | 133,24 / 100,00 | 117,61 / 89,60 | 183,81 / 142,00 |
| Whisper-large-v3 zero-shot | — | 92,50 / 63,20 | 99,74 / 76,71 | 94,00 / 67,90 |
| **Whisper-small fine-tuné** | 34,80 / 15,94 | 26,66 / 11,81 | **34,55 / 16,91** | **27,78 / 13,38** |
| Whisper-large-v3 fine-tuné | non réalisé | 22,72 / 9,77 | non réalisé | 25,19 / 12,33 |
| w2v-BERT-2.0 fine-tuné (meilleur du papier) | non réalisé | 19,92 / 7,10 | non réalisé | 21,37 / 8,34 |

- Notre Whisper-small fine-tuné est **6,8 points de WER au-dessus** de celui du papier sur le test (8,1 sur le dev).
  Différences connues : 80 epochs avec SpeechBrain dans le papier contre un early stopping sur la dev loss (meilleur
  checkpoint vers 5,6 epochs) avec transformers ; optimiseur Adam dans le papier, AdamW 8 bits et batch plafonné ici ;
  le papier ne décrit pas sa normalisation pour le WER (ici : ponctuation retirée, casse latine ignorée). Les zero-shot ne sont
  évalués que sur le test ici.

## 4. Traduction de la parole (tunisien → anglais)

BLEU au protocole du papier (sacreBLEU, casse gardée, ponctuation retirée) ; « sacreBLEU » = BLEU 13a standard ;
COMET = Unbabel/wmt22-comet-da (x 100), calculé sur le test seulement. « oracle » = traduction des transcriptions de
référence (borne haute d'une cascade).

### 4.1 Dev (730 segments ; sert aux choix)

| Système | type | BLEU [IC 95 %] | sacreBLEU | chrF++ |
|---|---|---|---|---|
| **C2** : whisper-small-ft → NLLB LoRA | pipeline | **21,05** [19,7 ; 22,4] | 20,83 | **45,29** |
| C3 : whisper-small-ft → NLLB LoRA (ASR bruité) | pipeline | 20,65 [19,2 ; 22,1] | 20,41 | 45,02 |
| **E2E-2** : Whisper-small (initialisé depuis l'ASR) | direct | **19,08** [17,4 ; 20,6] | 18,80 | **43,61** |
| C1 : whisper-small-ft → NLLB zero-shot | pipeline | 8,83 [7,7 ; 10,1] | 7,96 | 31,19 |
| C4 : whisper-small-ft → LLM (Qwen2.5-1.5B) | pipeline | 6,13 [5,2 ; 7,0] | 5,32 | 27,46 |

Choix sur le dev : meilleur direct = **E2E-2**, meilleur pipeline = **C2** (C3 n'est pas significativement
différent de C2 : -0,40 BLEU [-1,14 ; +0,24]).

### 4.2 Test (841 segments)

| Système | type | BLEU [IC 95 %] | sacreBLEU | chrF++ | COMET |
|---|---|---|---|---|---|
| C3 oracle : référence → NLLB LoRA (ASR bruité) | oracle | 31,98 [30,6 ; 33,4] | 31,75 | 55,55 | 75,46 |
| C2 oracle : référence → NLLB LoRA | oracle | 31,64 [30,3 ; 33,1] | 31,37 | 55,37 | 75,61 |
| **C2** : whisper-small-ft → NLLB LoRA (retenu, pipeline) | pipeline | **22,78** [21,2 ; 24,3] | 22,60 | **46,59** | **67,03** |
| C3 : whisper-small-ft → NLLB LoRA (ASR bruité) | pipeline | 22,82 [21,3 ; 24,3] | 22,68 | 46,52 | 66,89 |
| **E2E-2** (retenu, direct) | direct | **20,92** [19,5 ; 22,3] | 20,76 | **44,70** | **67,02** |
| C0 oracle : référence → NLLB zero-shot | oracle | 13,07 [11,8 ; 14,5] | 12,01 | 37,09 | 65,00 |
| C1 : whisper-small-ft → NLLB zero-shot | pipeline | 9,83 [8,8 ; 10,9] | 9,24 | 32,26 | 58,48 |
| C4 : whisper-small-ft → LLM (Qwen2.5-1.5B) | pipeline | 9,59 [8,4 ; 10,6] | 8,08 | 30,73 | 60,12 |
| B0 : Whisper-small zero-shot | direct | 4,40 [3,7 ; 5,1] | 3,55 | 22,63 | 53,36 |
| B0 : Whisper-large-v3 zero-shot | direct | **13,95 : chiffre du papier, non reproduit** | — | — | — |

- **Test de significativité apparié (bootstrap, 1 000 tirages), meilleur pipeline - meilleur direct choisis sur le
  dev (C2 - E2E-2)** : BLEU **+1,84** [+0,45 ; +3,15], p = 0,002 ; chrF++ **+1,88** [+0,99 ; +2,77], p < 0,001. La
  cascade C2 est **significativement meilleure** que la traduction directe en BLEU et en chrF++. En COMET, les deux
  sont à égalité (67,03 contre 67,02 ; pas de test apparié calculé pour COMET).
- **C3 = C2** sur le test aussi (22,82 contre 22,78 BLEU ; 46,52 contre 46,59 chrF++) : l'entraînement avec des
  sources ASR bruitées n'apporte rien, constat du dev confirmé.
- **Coût des erreurs d'ASR** : 8,9 points de BLEU entre C2 oracle (31,64) et C2 (22,78). Le fine-tuning de NLLB
  apporte 18,6 points sur les transcriptions de référence (C0 13,07 → C2 oracle 31,64) et 13,0 points sur la sortie
  ASR (C1 → C2).
- **LLM** : C4 (9,59) est au niveau de NLLB zero-shot (C1, 9,83) en BLEU, un peu au-dessus en COMET (60,12 contre
  58,48), très loin de NLLB fine-tuné.
- **Whisper-large-v3 zero-shot (B0)** : non reproduit. Sur le test (842 segments, beam 4, lots de 4, fp16, jusqu'à
  225 tokens générés), le décodage a tourné plus d'1 h 30 sans finir sur le GPU de 6 Go (mémoire saturée, 5,9 Go) ;
  cause probable : génération jusqu'à la limite de tokens, avec 16 séquences en beam sur un modèle de 1,55 B
  paramètres (non vérifié, aucune sortie écrite). Chiffre du papier cité : 10,96 dev / 13,95 test
  (CHANGELOG n° 58).

### 4.3 Analyse (test)

- **Propagation des erreurs d'ASR** (C2) : corrélation de Spearman entre le WER et le chrF du segment : -0,58.
  chrF moyen selon le WER du segment : 64,3 (WER ≤ 10 %, 151 segments), 52,4 (10-30 %, 264), 41,2 (30-60 %, 267),
  26,2 (> 60 %, 159).
- **Par type de segment** (chrF, test) : C2 38,7 (TUN) / 47,1 (MIXED) / 64,6 (FOR) ; E2E-2 38,3 / 44,9 / 61,9.
  L'avantage de la cascade vient surtout des segments mixtes et étrangers.
- **Types d'erreurs** (heuristiques automatiques, part des segments) : E2E-2 : correcte ou mineure 80,2 %,
  contresens 13,6 %, ajout / hallucination 4,4 %, omission 1,1 %, répétition en boucle 0,5 % ; C2 : 80,6 / 13,2 /
  3,3 / 2,4 / 0,2 %. Profils très proches ; la cascade hallucine un peu moins, omet un peu plus.
- **Autres langues cibles** (sans référence dans TEDxTN : mesure indicative par aller-retour vers l'anglais avec
  NLLB, chrF++ contre la référence anglaise) : français 42,2, arabe standard 39,9.

### 4.4 Comparaison avec le papier (Tableau 7, BLEU au même protocole)

| Système | dev (ici) | dev (papier) | test (ici) | test (papier) |
|---|---|---|---|---|
| Whisper-small zero-shot | — | 3,98 | 4,40 | 5,70 |
| Whisper-large-v3 zero-shot | non reproduit | 10,96 | non reproduit | 13,95 |
| **Whisper-small fine-tuné** (ici : E2E-2, initialisé depuis l'ASR) | 19,08 | 17,31 | **20,92** | **18,53** |
| Whisper-medium fine-tuné | non réalisé | 23,02 | non réalisé | 24,50 |
| Whisper-large-v3 fine-tuné (meilleur du papier) | non réalisé | 25,19 | non réalisé | 25,68 |
| **Cascade C2** (whisper-small-ft → NLLB-600M LoRA ; pas d'équivalent dans le papier) | 21,05 | — | **22,78** | — |

- À taille égale (Whisper-small), notre traduction directe dépasse le papier de **+2,4 BLEU** sur le test (+1,8 sur
  le dev), alors que notre ASR whisper-small est moins bon que le sien : l'initialisation depuis l'ASR fine-tuné en
  est une explication possible, mais elle n'est pas isolée (pas d'E2E-1). La cascade C2, avec des modèles de moins
  de 1 B paramètres chacun, reste 2,9 BLEU sous le meilleur système du papier (Whisper-large-v3 fine-tuné, 1,55 B).
  Le papier rapporte qu'un encodeur w2v-BERT / XLSR relié à NLLB-1.3B n'a pas dépassé 5 BLEU ; il n'évalue pas de
  cascade ASR → NLLB.

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
  modèle portent donc sur le fond (mots, emprunts, segmentation) bien plus que sur l'orthographe. Confirmé sur le
  test (§ 3.1 : 0,24 point). (Le chiffre de 34,57 % cité avant la décision venait d'une configuration intermédiaire : ancienne
  `EVAL_NORM` sans les hésitations mais avec la frontière de script.)

### ASR retenu pour les cascades

- **asr_best = whisper-small-ft, définitivement** (seul ASR fine-tuné du projet ; les baselines zero-shot ne sont évaluées que
  sur le test, comme dans le papier). Les cascades
  C1 à C4 et la post-correction par LLM partent toutes de cette transcription.

### Post-correction par LLM et source de C4 (règle fixée avant le calcul, appliquée sur le dev)

- La post-correction de l'ASR par LLM (Qwen2.5-1.5B, exemples few-shot du train) n'est **retenue** que si elle baisse
  le WER brut du dev de façon significative : IC 95 % du bootstrap apparié (1 000 tirages) de la différence de WER
  (corrigé - d'origine) **entièrement sous 0** (`metrics.correction_helps`, la même fonction dans les notebooks 01 et 02).
- **C4** (traduction par LLM) traduit la transcription corrigée seulement si la correction est retenue ; sinon il
  traduit la **transcription ASR d'origine** (même source pour le dev et le test ; le nom du système l'indique :
  `C4-[whisper-small-ft+llm]->llm` ou `C4-[whisper-small-ft]->llm`).
- **Raison** : une correction qui n'améliore pas la transcription de façon démontrée sur le dev peut introduire des
  erreurs (mots changés, hallucinations) que la traduction propage ; en la gardant, l'écart entre C4 et les autres
  cascades mélangerait l'effet du traducteur LLM et celui d'une correction non validée. La décision est prise sur le
  dev uniquement ; le test ne sert qu'à rapporter.
- **Résultat sur le dev (whisper-small-ft, 731 segments) : correction non retenue.** WER brut 34,80 % → 42,51 %
  (normalisé 34,54 → 41,60 ; sans hésitations 35,16 → 41,93 ; CER brut 15,94 → 21,48). Différence de WER brut
  +7,72 points, IC 95 % apparié [+6,72 ; +8,71] : la correction **dégrade** nettement la transcription. Le LLM modifie
  75,2 % des segments (toutes ses sorties passent le filtre de similarité) et en raccourcit 30 % (-0,42 mot en moyenne).
  Exemples observés : mots corrects remplacés (`هذاكا` → `الحالي`, plus proche de l'arabe standard), emprunt français
  bien reconnu remplacé par un mot arabe inexistant (`dépression` → `دهقنة`), français « corrigé » à tort
  (`vos profilés` → `vous profils`), `و` ajouté à une phrase juste. Un LLM de 1,5 B paramètres (quantifié en 4 bits)
  ne connaît pas assez le tunisien ni le code-switching pour corriger l'ASR. **C4 traduit donc la transcription ASR
  d'origine.** Confirmé sur le test (§ 3.1 : +8,17 points de WER).
- **Pourquoi le garde-fou (similarité ≥ 0,6) a laissé passer 550 modifications sur 731** (constat d'analyse ; le seuil
  n'a pas été changé après coup) :
  1. il mesure la **ressemblance avec l'hypothèse ASR**, pas la justesse : il écarte les réécritures complètes,
     traductions et hallucinations, pas les petites modifications fausses ;
  2. il est calculé **en caractères** (`SequenceMatcher`) : changer un ou deux mots d'une phrase de quinze mots garde
     une similarité d'environ 0,9. Sur les 550 sorties acceptées : médiane 0,95, 10e centile 0,80, 72,2 % ≥ 0,9 ;
     le seuil de 0,6 n'est presque jamais atteint ;
  3. les 6 exemples few-shot ont été choisis parmi les segments du train où l'ASR se trompe (similarité 0,83 à 0,99,
     WER 6 à 41 %) : le modèle ne voit que des exemples où **il faut changer quelque chose**, jamais un cas où la
     transcription est à recopier telle quelle ;
  4. la consigne demande d'écrire les emprunts français / anglais en latin, ce qui pousse à toucher aux mots latins :
     sur les 363 segments acceptés dont l'hypothèse contient du latin, le LLM a modifié les mots latins dans 229.
  Effet par segment (WER du segment) sur les 550 sorties acceptées : 70,0 % dégradées, 26,5 % inchangées, 3,5 %
  améliorées. Erreurs : substitutions 25,9 → 31,5 %, suppressions 3,7 → 5,8 %, insertions 5,2 → 5,2 %. Limite de la
  mesure : les sorties brutes du LLM ne sont pas sauvegardées, on ne peut donc pas distinguer, parmi les 181 segments
  non modifiés, ceux que le garde-fou a rejetés de ceux que le LLM a recopiés.

### C3 (NLLB LoRA entraîné avec 50 % de sorties ASR bruitées) : constat sur le dev

- Entraînement complet (6 epochs, 2940 steps, sans early stopping), modèle retenu : checkpoint-2800 (dev loss 2,8902
  contre 2,8879 pour C2 ; courbes presque superposées).
- Dev (730 segments, source whisper-small-ft) : BLEU 20,65 [19,07 ; 22,19], chrF++ 45,02 [43,78 ; 46,26], contre
  21,05 / 45,29 pour C2. Bootstrap apparié C3 - C2 : BLEU -0,40 [-1,14 ; +0,24] (p = 0,89), chrF++ -0,28
  [-0,73 ; +0,14] (p = 0,90) : **pas de différence significative**. L'entraînement sur sorties ASR bruitées n'a pas
  rendu NLLB plus robuste aux erreurs de whisper-small-ft. Confirmé sur le test (§ 4.2 : 22,82 contre 22,78 BLEU).

### Améliorations envisagées, non réalisées (ressources de calcul)

- **w2v-BERT-2.0 CTC (+ LM 4-grammes)** : le fine-tuning complet (~600 M paramètres) ne tient pas sur le GPU local de
  6 Go ; il était prévu sur Colab (notebook 01b) et a été abandonné. La comparaison des systèmes ASR avec le papier
  se fait donc sans ce modèle.
- **E2E-1** (Whisper-small en traduction, initialisé depuis le modèle de base) : abandonné faute de temps GPU (5 à 7 h
  sur le GPU local). **Conséquence** : sans E2E-1, l'effet de l'initialisation depuis l'ASR fine-tuné (E2E-2) n'est
  pas isolé ; on ne peut pas dire quelle part du score de l'E2E-2 vient de cette initialisation.
- **Post-correction LLM** : un LLM plus gros (Qwen2.5-7B-Instruct, prévu par le code au-delà de 20 Go de VRAM,
  impossible sur 6 Go), ou un LLM fine-tuné sur des paires tunisiennes (sortie ASR, référence) du train ; une
  consigne plus stricte (ne jamais transformer un mot en écriture latine, recopier tel quel en cas de doute) et des
  exemples few-shot qui incluent des transcriptions justes à recopier sans changement.

### Écarts au protocole du papier

- Train : 15 703 segments dans `train.csv` du Hub contre 15 626 dans le papier (+77), gardés.
- Outils et durée : transformers (et non SpeechBrain), early stopping sur la dev loss au lieu de 80 epochs ;
  AdamW 8 bits au lieu d'Adam ; batch par GPU plafonné à 4 avec accumulation.
- Modèles : seulement Whisper-small (ASR et traduction directe) ; pas de Whisper-medium / large-v3 fine-tunés, ni de
  MMS, XLSR, w2v-BERT. Whisper-large-v3 en traduction zero-shot non reproduit (chiffre du papier cité).
- NLLB-600M fine-tuné en LoRA (r = 16 sur q/k/v/out_proj et fc1/fc2, poids fp32, précision mixte fp16) au lieu d'un
  fine-tuning complet (GPU de 6 Go). Les cascades (C0-C4) n'existent pas dans le papier.
- Ajouts : COMET, chrF++, IC 95 %, test apparié, post-correction et traduction par LLM, WER normalisé / sans
  hésitations / script-aware, analyse par talk.

## Bugs rencontrés (détail : CHANGELOG_EXECUTION.md, n° 1 à 58)

Ceux qui touchent l'interprétation des résultats :
- n° 46 : courbe du WER de suivi de l'ASR perdue pour les steps 250 à 750 (arrêt volontaire) ; recalculée pour 1000
  et 1250. Sans effet sur le modèle retenu (choix sur la dev loss).
- n° 48 à 50 : normalisation d'évaluation corrigée **avant** tout calcul sur le test (WER normalisé qui montait,
  hésitations mal repérées, « أم » retiré à tort).
- n° 51 : nom des cascades sans l'ASR source ; corrigé avant les cascades, avec refus de relire des prédictions
  faites avec un autre ASR.
- n° 55 : C4 aurait traduit la transcription corrigée par LLM même non retenue ; corrigé avant C4.
- n° 58 : baseline Whisper-large-v3 zero-shot (traduction) trop lente sur 6 Go, abandonnée pendant la passe finale.
- Aucun bug de calcul de score n'a été trouvé pendant la passe finale ; aucun système n'a été modifié après le test.

## Limites

- **Un seul entraînement par système** (une graine) : la variance entre essais n'est pas mesurée ; les IC ne
  reflètent que l'échantillonnage des segments.
- **Dev et test petits** (5 et 6 talks) : les IC sont larges (environ ± 2 à 3 points de WER, ± 1,5 BLEU) et un talk
  peut peser lourd ; les écarts de moins d'un point ne sont pas interprétables sans test apparié.
- **Choix du checkpoint sur la dev loss** : bien adapté à l'ASR, moins à la traduction (voir E2E-2).
- **Comparaison au papier imparfaite** : outils, durée d'entraînement, normalisation du WER (non décrite dans le
  papier) et définition du CMI différents ; seuls les modèles de taille « small » sont comparables.
- **COMET** : un seul modèle (wmt22-comet-da), calculé sur le test seulement ; pas de test apparié en COMET.
- **Catégories d'erreurs automatiques** (ASR et traduction) : heuristiques, non validées à la main.
- **Français et arabe standard** : pas de référence dans TEDxTN, mesure par aller-retour seulement.
- **Train bruité conservé** : 6 talks TEDx Pasha Street très bruités (SNR ≈ 4 dB, WER ≈ 100 %) restent dans le train.
- **Beam 5** : évalué sur le test seulement, donc pas sélectionnable ; le gain observé (-2,5 WER) reste à confirmer
  sur le dev.

## À faire

- 13/11 : ByT5 (1 epoch, contrôle de recopie avant tout score) et passe finale du notebook 03 (section 6 de ce
  document : normalisation) ; remise des réglages de veille d'origine.
- Hors du temps de calcul disponible : voir « Améliorations envisagées » (w2v-BERT, E2E-1, Whisper plus grands,
  LLM plus gros ou fine-tuné, beam 5 évalué sur le dev, règle de choix du checkpoint sur le BLEU du dev pour la
  traduction).
