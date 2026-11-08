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
| 3 | `common/data.py` (`discover_urls_in_repo`) | Le dépôt HF fournit bien `TEDxTN_URLS.txt` (108 lignes « URL nom_du_talk »), mais le nom était cherché en découpant sur `[,\t;\|]` seulement : avec un séparateur espace, aucun talk n'était associé et les 108 talks partaient en recherche YouTube par titre (1 mauvaise vidéo dès le premier essai : un reportage TV5 de 65 s pour un talk de 18 min). | Découpage aussi sur les blancs. |
| 4 | `common/data.py` (`VIDEO_URL_RE`) | 2 talks de TEDxCarthage (Nabil Karoui, Tarek Cheniti) ont une URL **Facebook** officielle, ignorée par une regex limitée à YouTube. | Regex étendue (YouTube, Facebook, Vimeo, Dailymotion) ; yt-dlp les télécharge. |
| 5 | `common/data.py` (`build_url_map`, `check_video`, `video_info`, `search_youtube`), `scripts/prepare_data.py --check-urls` | Aucun contrôle des URL avant téléchargement, sauf la durée après coup. | Contrôle avant téléchargement : durée ≥ dernière annotation, chaîne TEDx, événement et orateur dans le titre (comparaison aussi sans espaces : « TEDxPasha Street »), similarité du titre. Statut OK / douteux / introuvable + motifs dans `youtube_urls.csv`. La recherche par titre garde le meilleur des 5 premiers résultats. |
| 6 | `common/data.py` (`_yt_dlp`, `video_info`) | Après ~200 requêtes à 8 en parallèle, YouTube renvoie HTTP 429 puis « Sign in to confirm you're not a bot » : tous les talks étaient alors marqués « introuvable » à tort. | Exception `YouTubeBlocked` distincte de « vidéo inexistante » (le talk reste non vérifié), 2 requêtes en parallèle, `--sleep-requests` (`YTDLP_SLEEP`, défaut 1 s), cookies via `YTDLP_COOKIES`. |
| 7 | `common/data.py` (`cut_segments`, `fetch_all_audio`) | Pas de moyen de corriger un décalage entre la vidéo et les annotations, ni d'exclure un talk. | Colonne `offset_s` (temps audio = temps annoté + offset) appliquée au découpage (offset inscrit dans le nom des fichiers pour forcer le redécoupage) ; statuts `exclu` / `introuvable` ignorés au téléchargement et au découpage. |
| 8 | `common/data.py` (`_yt_dlp`, `fetch_all_audio`), `scripts/prepare_data.py` | Options anti-blocage codées en dur (une seule pause), 4 téléchargements en parallèle, aucune reprise après un 429. | Options configurables (`YTDLP_SLEEP_REQUESTS`=1, `YTDLP_SLEEP_INTERVAL`=3, `YTDLP_MAX_SLEEP_INTERVAL`=8, `YTDLP_MAX_PARALLEL`=2, `YTDLP_BLOCK_WAIT`=900 s, `YTDLP_BLOCK_RETRIES`=4) ; en cas de blocage, pause partagée par tous les threads puis reprise de la même requête ; les talks déjà téléchargés sont sautés. Runtime JS `deno` (pip) installé pour yt-dlp. |
| 9 | `notebooks/01` (§9, §10, §13), `notebooks/02` (§7.2, §8, §9, §11, §13) | **Choix faits sur le test** : système analysé/exporté du 01 repli sur le classement test sans prédiction dev ; dans le 02, meilleur direct / meilleur pipeline, comparaison appariée et export `ast_best` triés sur le chrF++ du **test** ; post-correction LLM (01) et traduction LLM C4 (02) évaluées sur le test seulement. | Tous les choix sur le dev : tableau `ast_results_dev.csv`, fonction `best_on_dev()`, colonne « retenu (dev) » dans `ast_results.csv` ; LLM évalué sur dev et test, retenu seulement s'il baisse le WER du dev ; `BEST` = meilleur WER normalisé du dev, sinon aucun. |
| 10 | `notebooks/01` §2, `notebooks/02` §2, `common/config.py` | Interrupteurs `RUN_*` = « GPU présent » : w2v-BERT complet, SeamlessM4T (2,3 B) et Qwen2.5-7B lancés sur un GPU de 6 Go (OOM) ; batch d'inférence fixe (16). | `env_flag()` (forçage `TEDXTN_RUN_<NOM>`), `gpu_memory_gb()` : CTC complet et Seamless seulement si VRAM ≥ 12 Go, `LLM_MODEL` = 7B si ≥ 20 Go sinon 1.5B, `GEN_BS` = 16 si ≥ 12 Go sinon 4. |
| 11 | `common/asr.py` (`training_args`) | Batch codé en dur dans les notebooks (16, 8) ; impossible de faire un essai court. | `TEDXTN_MAX_BATCH` plafonne le batch et augmente l'accumulation d'autant (batch effectif inchangé) ; `TEDXTN_MAX_STEPS`, `TEDXTN_EVAL_STEPS` ; `logging_steps` adapté. |
| 12 | `common/config.py` (`SMOKE`), `notebooks/01` §4.2, `notebooks/02` §3 | Pas de mode d'essai rapide. | `TEDXTN_SMOKE=1` : 200 segments de train, pseudo-dev et pseudo-test = deux talks de **train** tenus à part (dev et test officiels non utilisés). |
| 13 | `common/asr.py` (`load_whisper`) | En inférence, le modèle était chargé en **fp32 sur le GPU** puis converti par `.half()` dans le notebook : whisper-large-v3 (1,55 B param.) transite à 6,2 Go -> « CUDA driver error: out of memory » au chargement (smoke test). | Chargement direct en fp16 si `for_training=False` et CUDA disponible. En entraînement, les poids restent en fp32 (vérifié : `torch.float32`) et `fp16=True` fait la précision mixte ; `load_ctc` ne change pas le dtype. |
| 14 | `common/asr.py` (`whisper_generate`) | Avec 5,0 Go réellement libres (Windows/WDDM en garde ~1 Go), `generate` de whisper-large-v3 en batch 4 sature la VRAM sur de vrais segments (pic mesuré 4,74 Go sur de l'audio court) : toute l'évaluation échoue. | Repli automatique : sur « out of memory », le lot est coupé en deux et retraité (jusqu'à 1). Post-traitement isolé dans `_postprocess`. |
| 15 | `notebooks/01` §5 | Après une erreur de génération, le modèle restait en VRAM : la cellule suivante (Whisper FT) échouait à son tour (`INTERNAL ASSERT FAILED ... CUDACachingAllocator`). | `try/finally` : `del model; torch.cuda.empty_cache()` même en cas d'erreur. |
| 16 | `notebooks/01` §13 | Sans prédiction dev, la conclusion affichait « Système retenu (choisi sur le dev) » avec le premier système du test. | Conclusion chiffrée seulement pour le système `BEST` choisi sur le dev ; sinon message explicite. |
| 17 | Machine (constat, pas de code) | w2v-BERT-2.0 en fine-tuning complet : sous WSL/WDDM, le dépassement de VRAM ne lève pas d'OOM mais déborde en mémoire partagée -> la cellule tourne à 0 % de GPU sans fin (constaté pendant 10 min). | Entraînement w2v-BERT sur Colab (notebook 01b) ; en local `RUN_CTC_FT` vaut False sous 12 Go. |
| 18 | `scripts/build_kenlm.sh` | `apt-get update` échoue dès qu'un dépôt tiers est mal signé (dépôt Jenkins sur la machine locale) ; avec `set -e`, le script s'arrêtait (code 100) avant de compiler KenLM. | apt seulement si cmake/boost manquent ; échec de `apt-get update` non bloquant. |
| 19 | Environnement d'exécution (constat) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` sous WSL : « INTERNAL ASSERT FAILED (CUDACachingAllocator.cpp:400) » dès que la VRAM se remplit. | Option non utilisée en local (gardée dans 01b : Colab est un Linux natif). |
## Constats sur les données (aucune modification des splits)

- Train : `train.csv` du Hub contient **15 703 segments** contre 15 626 dans le papier (+77) ; dev (731) et test (842)
  correspondent exactement. Pas de doublon ni de bug de fusion : l'écart vient du fichier publié. 1 transcription vide
  et 2 segments sans traduction dans le train. Heures : train 22,67 h, dev 1,13 h, test 1,24 h (papier : 22h40, 1h07, 1h14).
- Aucune fuite : aucun talk partagé entre splits, aucune transcription longue (≥ 6 mots) partagée. Un seul
  chevauchement de 36 ms (train, talk Fakrouna_Rising…), négligeable.
