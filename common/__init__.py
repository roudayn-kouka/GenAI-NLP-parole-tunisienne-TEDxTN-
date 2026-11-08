"""Code partagé du projet TEDxTN (ASR + traduction de la parole tunisienne).

Modules :
    config     chemins, modèles, langues cibles, hyperparamètres
    data       annotations, audio YouTube, découpage, contrôle des fuites
    text       script, code-switching, CMI, normalisation, comparaison inter-scripts
    metrics    WER/CER bruts et normalisés, BLEU, chrF++, COMET, bootstrap, analyse d'erreurs
    augment    augmentation audio (vitesse, bruit, gain)
    asr        Whisper (complet / LoRA) et CTC (w2v-BERT-2.0, MMS, XLSR) + KenLM
    translate  NLLB, SeamlessM4T, encodeur parole + décodeur texte
    llm        post-correction, traduction et normalisation par LLM (volet GenAI)
    pipeline   pipeline complet avec trace étape par étape (utilisé par l'interface)
"""
