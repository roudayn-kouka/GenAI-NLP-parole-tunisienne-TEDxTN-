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

## Constats sur les données (aucune modification des splits)

- Train : `train.csv` du Hub contient **15 703 segments** contre 15 626 dans le papier (+77) ; dev (731) et test (842)
  correspondent exactement. Pas de doublon ni de bug de fusion : l'écart vient du fichier publié. 1 transcription vide
  et 2 segments sans traduction dans le train. Heures : train 22,67 h, dev 1,13 h, test 1,24 h (papier : 22h40, 1h07, 1h14).
- Aucune fuite : aucun talk partagé entre splits, aucune transcription longue (≥ 6 mots) partagée. Un seul
  chevauchement de 36 ms (train, talk Fakrouna_Rising…), négligeable.
