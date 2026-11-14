"""Contenu de l'onglet « Résultats des expériences » : tableaux et figures de results/, chacun suivi d'un court
paragraphe (ce qu'il montre, comment le lire, constat). Les chiffres des paragraphes sont lus dans results/*.csv au
moment de l'affichage (rien n'est écrit en dur) ; ceux du papier viennent de common/config.py et sont signalés.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from common.config import PAPER_ASR_RESULTS, PAPER_AST_RESULTS, RESULTS_DIR

PAPER = "<span class='dv-paper'>papier</span>"   # étiquette des chiffres repris du papier


@dataclass
class Item:
    kind: str          # "table" | "image"
    title: str
    payload: object    # DataFrame ou chemin d'image
    caption: str       # HTML (paragraphe sous l'élément)


def _csv(name: str, root: Path) -> pd.DataFrame | None:
    p = Path(root) / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _img(name: str, root: Path) -> str | None:
    p = Path(root) / f"{name}.png"
    return str(p) if p.exists() else None


def fr(x, nd=2) -> str:
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{nd}f}".replace(".", ",")


def _row(df, col, val):
    r = df[df[col] == val]
    return r.iloc[0] if len(r) else None


# --------------------------------------------------------------------------
def asr_items(root=RESULTS_DIR) -> list[Item]:
    out = []
    asr = _csv("asr_results", root)
    if asr is not None:
        ft = _row(asr, "système", "whisper-small-ft")
        zs = asr[asr["système"].str.contains("zeroshot")].sort_values("WER")
        llm = _row(asr, "système", "whisper-small-ft+llm")
        b5 = _row(asr, "système", "whisper-small-ft-beam5")
        paper = PAPER_ASR_RESULTS["Whisper-small (FT)"]
        cap = ("<b>Ce que montre le tableau</b> : la reconnaissance sur le <b>test</b> (842 segments). WER / CER = part de "
               "mots / caractères à corriger pour retrouver la référence : <b>plus bas = mieux</b> (au-delà de 100 % : la "
               "sortie contient plus d'erreurs que la référence n'a de mots). IC 95 % : bootstrap sur les segments. "
               "« normalisé » ignore les variantes d'orthographe, « sans hésitations » retire « آ, euh... ».")
        if ft is not None:
            cap += (f"<br><b>Constat</b> : le système retenu sur le dev, whisper-small-ft, fait <b>{fr(ft['WER'])} %</b> de WER "
                    f"{ft['IC95 WER']} et {fr(ft['CER'])} % de CER, contre {fr(paper['WER'])} % / {fr(paper['CER'])} % pour le "
                    f"même modèle dans le {PAPER} (entraîné plus longtemps).")
            if len(zs):
                cap += f" Le fine-tuning retire {fr(zs.iloc[0]['WER'] - ft['WER'], 1)} points au meilleur zero-shot."
        if b5 is not None:
            cap += (f" Le beam 5 ({fr(b5['WER'])} %) n'a été décodé que sur le test : il ne pouvait pas être choisi "
                    "(choix sur le dev).")
        if llm is not None and ft is not None:
            cap += f" La post-correction par LLM <b>dégrade</b> le WER (+{fr(llm['WER'] - ft['WER'])} points)."
        out.append(Item("table", "Reconnaissance sur le test (WER / CER)", asr, cap))
        pap = pd.DataFrame([{"système (papier)": k, "WER test": v["WER"], "CER test": v["CER"]}
                            for k, v in PAPER_ASR_RESULTS.items()])
        out.append(Item("table", "Repères du papier (Tableau 6, test)", pap,
                        f"Chiffres du {PAPER} (Bougares et al., 2025), non recalculés ici : 80 epochs avec SpeechBrain, "
                        "sans LM. Seul Whisper-small est comparable à notre système ; w2v-BERT-2.0 (le meilleur) n'a pas "
                        "pu être entraîné sur le GPU de 6 Go."))
    p = _img("01_wer_comparison", root)
    if p and asr is not None:
        best = asr.sort_values("WER").iloc[0]
        out.append(Item("image", "WER par système (test)", p,
                        "Barres : WER sur le test de chaque système (plus court = mieux), repères du papier en "
                        f"pointillés. Le meilleur WER du graphique est {best['système']} ({fr(best['WER'])} %)."))
    curve = _csv("dev_wer_curve_asr_whisper-small", root)
    log = _csv("train_log_asr_whisper-small", root)
    p = _img("01_curve_train_log_asr_whisper-small", root)
    if p:
        cap = ("Courbes d'entraînement de l'ASR sur le <b>dev</b> : la loss (plus bas = mieux) sert à l'early stopping et "
               "au choix du checkpoint ; le WER est suivi sur 200 segments du dev.")
        if log is not None and "eval_loss" in log:
            ev = log.dropna(subset=["eval_loss"])
            b = ev.loc[ev.eval_loss.idxmin()]
            cap += (f" Meilleure loss au step {int(b.step)} ({fr(b.eval_loss, 4)}) : c'est le checkpoint retenu ; "
                    f"arrêt au step {int(ev.step.max())}.")
        out.append(Item("image", "Entraînement de l'ASR (dev)", p, cap))
    for name, title, cap in [
        ("01_error_categories", "Catégories d'erreurs (test)",
         "Répartition des erreurs du système retenu selon leur nature (catégories automatiques, heuristiques) : "
         "mots proches à l'oreille, erreurs lexicales, emprunts / code-switching, insertions, suppressions, même mot "
         "écrit dans l'autre écriture. Les erreurs de fond dominent ; les variantes d'orthographe pèsent peu."),
        ("01_error_breakdown", "WER selon la durée, le mélange de langues et le talk (test)",
         "WER moyen du système retenu selon la durée du segment (secondes), l'indice de mélange des langues (CMI) et "
         "le talk. Plus la barre est courte, mieux c'est. Les segments courts sont les plus difficiles (peu de "
         "contexte) ; l'écart entre talks est plus grand que l'effet du mélange de langues."),
        ("01_distributions", "Le corpus : durée, mots et débit par split",
         "Trois graphiques, avec dans chacun un histogramme par split (train, dev, test ; normalisés, donc "
         "comparables malgré des tailles différentes) : durée des segments (secondes), nombre de mots par segment et "
         "débit de parole (caractères de la transcription par seconde). Les trois splits ont des distributions proches : le dev et le test "
         "ressemblent au train sur ces trois mesures.")]:
        p = _img(name, root)
        if p:
            if name == "01_distributions":
                h = _csv("data_hours", root)
                if h is not None:
                    cap += " " + " ; ".join(f"{r.split} : {int(r.segments)} segments, {fr(r.heures_utilisables)} h"
                                            for r in h.itertuples()) + "."
            out.append(Item("image", title, p, cap))
    hard = _csv("01_hard_talks", root)
    if hard is not None and {"WER", "SNR estimé (dB)"} <= set(hard.columns):
        rho = hard[["WER", "SNR estimé (dB)"]].corr(method="spearman").iloc[0, 1]
        cols = [c for c in ["talk", "split", "WER", "CER", "SNR estimé (dB)", "CMI", "mots latins (%)"] if c in hard]
        show = hard.sort_values("WER", ascending=False).head(8)[cols].round(1)
        out.append(Item("table", "Talks difficiles (WER par talk)", show,
                        "WER et caractéristiques par talk (dev, test et talks du train signalés). Corrélation de "
                        f"Spearman entre le WER du talk et le rapport signal / bruit estimé : <b>{fr(rho)}</b> (sur "
                        f"{len(hard)} talks) : plus l'enregistrement est bruité, plus le WER monte."))
    return out


# --------------------------------------------------------------------------
def ast_items(root=RESULTS_DIR) -> list[Item]:
    out = []
    dev, test = _csv("ast_results_dev", root), _csv("ast_results", root)
    if dev is not None:
        bd, bp = dev[dev.type == "direct"].iloc[0], dev[dev.type == "pipeline"].iloc[0]
        out.append(Item("table", "Traduction sur le dev (sert aux choix)", dev.drop(columns=["COMET"], errors="ignore"),
                        "Traduction tunisien → anglais sur le <b>dev</b> (730 segments). BLEU (protocole du papier : casse "
                        "gardée, ponctuation retirée) et chrF++ : <b>plus haut = mieux</b>. « pipeline » = ASR puis "
                        "traduction du texte ; « direct » = l'audio va directement vers l'anglais. "
                        f"<br><b>Choix</b> : meilleur direct = {bd['système']} (BLEU {fr(bd.BLEU)}), meilleur pipeline = "
                        f"{bp['système']} (BLEU {fr(bp.BLEU)})."))
    if test is not None:
        ret = test[test["retenu (dev)"] == True]   # noqa: E712
        cap = ("Mêmes systèmes sur le <b>test</b> (841 segments), évalué une seule fois. COMET (x 100) estime la "
               "qualité du sens avec un modèle neuronal : plus haut = mieux. « oracle » = traduction des "
               "transcriptions de référence : borne haute d'une cascade.")
        if len(ret):
            cap += "<br><b>Systèmes retenus sur le dev</b> : " + " ; ".join(
                f"{r['système']} : BLEU {fr(r.BLEU)} {r['IC95 BLEU']}, chrF++ {fr(r['chrF++'])}, COMET {fr(r.COMET)}"
                for _, r in ret.iterrows()) + "."
        c2r = test[(test.type == "oracle") & test["système"].str.contains("nllb-lora$")]
        c2 = ret[ret.type == "pipeline"]
        if len(c2r) and len(c2):
            cap += (f" Les erreurs de l'ASR coûtent {fr(c2r.iloc[0].BLEU - c2.iloc[0].BLEU, 1)} points de BLEU "
                    "(oracle contre sortie ASR).")
        out.append(Item("table", "Traduction sur le test", test, cap))
    pt = _csv("ast_paired_test", root)
    if pt is not None:
        b = pt.iloc[0]
        sig = "significativement meilleur" if b["IC95 bas"] > 0 else "pas significativement différent"
        out.append(Item("table", "Test apparié : meilleur pipeline contre meilleur direct (test)", pt,
                        "Bootstrap apparié (1 000 tirages sur les segments) de la différence de score B - A. Si "
                        "l'intervalle ne contient pas 0, l'écart est significatif. <b>Constat</b> : le pipeline est "
                        f"{sig} en BLEU ({'+' if b['B - A'] >= 0 else ''}{fr(b['B - A'])} [{fr(b['IC95 bas'])} ; "
                        f"{fr(b['IC95 haut'])}], p = {fr(b.p, 3)})."))
    pap = pd.DataFrame([{"système (papier)": k, "BLEU test": v} for k, v in PAPER_AST_RESULTS.items()])
    cap = (f"Chiffres du {PAPER} (Tableau 7, même protocole de BLEU). Whisper-large-v3 zero-shot n'a pas été reproduit "
           "ici (décodage trop lent sur 6 Go) : son chiffre est celui du papier.")
    if test is not None:
        e2e = test[test["système"].str.startswith("E2E")]
        if len(e2e):
            cap += (f" Notre traduction directe Whisper-small ({fr(e2e.iloc[0].BLEU)}) est à comparer à « Whisper-small "
                    f"(FT) » du papier ({fr(PAPER_AST_RESULTS['Whisper-small (FT)'])}).")
    out.append(Item("table", "Repères du papier (Tableau 7, test)", pap, cap))
    p = _img("02_ast_comparison", root)
    if p:
        cap = ("Longueur de la barre : <b>chrF++</b> de chaque système sur le test (plus long = mieux) ; le nombre à "
               "droite de la barre est son <b>BLEU</b>. Orange : traduction directe (un seul modèle, de l'audio à "
               "l'anglais) ; vert : pipeline (ASR puis traduction du texte). Les traductions des transcriptions de "
               "référence (oracle) et les chiffres du papier ne figurent pas sur cette figure (voir les tableaux "
               "ci-dessus).")
        if test is not None and "chrF++" in test:
            f = test[test.type != "oracle"].sort_values("chrF++", ascending=False)
            if len(f) >= 2:
                hi, lo = f.iloc[0], f.iloc[-1]
                cap += (f" <b>Constat</b> : de {lo['système']} (chrF++ {fr(lo['chrF++'])}, BLEU {fr(lo.BLEU)}) à "
                        f"{hi['système']} (chrF++ {fr(hi['chrF++'])}, BLEU {fr(hi.BLEU)}). Les cascades avec NLLB "
                        "fine-tuné (C2, C3) et la traduction directe E2E-2 dominent ; NLLB sans fine-tuning (C1) et "
                        "le LLM (C4) restent loin derrière.")
        out.append(Item("image", "chrF++ et BLEU par système (test)", p, cap))
    p = _img("02_error_propagation", root)
    if p:
        cap = ("Chaque point est un segment du test : en abscisse le WER de sa transcription, en ordonnée le chrF de sa "
               "traduction en cascade. Plus l'ASR se trompe, plus la traduction se dégrade.")
        rho = _propagation(root)
        if rho is not None:
            cap += f" Corrélation de Spearman WER / chrF calculée sur les prédictions : <b>{fr(rho)}</b>."
        out.append(Item("image", "Propagation des erreurs de l'ASR (test)", p, cap))
    return out


def _propagation(root) -> float | None:
    """Corrélation WER du segment / chrF de la traduction C2, calculée sur les prédictions du test."""
    try:
        from common import data, metrics
        dev = pd.read_csv(Path(root) / "ast_results_dev.csv")
        pipe = dev[dev.type == "pipeline"].iloc[0]["système"]
        asr = pd.read_csv(data.pred_path("asr", "whisper-small-ft", "test")).fillna("")
        tr = pd.read_csv(data.pred_path("ast", pipe, "test")).fillna("")
        m = asr.merge(tr, on="seg_id", suffixes=("_asr", "_tr"))
        w = [metrics.segment_wer(r, h) for r, h in zip(m.ref_asr, m.hyp_asr)]
        c = [metrics.sentence_chrf(h, r) for h, r in zip(m.hyp_tr, m.ref_tr)]
        return float(pd.Series(w).corr(pd.Series(c), method="spearman"))
    except Exception:
        return None


# --------------------------------------------------------------------------
def norm_items(root=RESULTS_DIR) -> list[Item]:
    out = []
    n = _csv("normalization_results", root)
    if n is not None:
        v = dict(zip(n["évaluation"], n["valeur"]))
        g = lambda k: next((val for key, val in v.items() if k in key), None)   # noqa: E731
        cap = ("Mesures de la normalisation (dev + test). « exactitude mots inconnus » : part des mots dont la graphie "
               "n'a jamais été vue à l'entraînement, ramenés à la forme majoritaire (plus haut = mieux). OOV : mots "
               "absents du vocabulaire du train (plus bas = mieux).")
        if g("ByT5") is not None:
            cap += (f"<br><b>Constat</b> : ByT5 ramène <b>{fr(g('ByT5'))} %</b> des variantes jamais vues à la forme "
                    f"majoritaire, contre {fr(g('lexique majoritaire'))} % pour le lexique (qui ne connaît pas ces mots).")
        if g("WER brut") is not None and g("WER normalisé") is not None:
            cap += (f" Sur le WER du test, la normalisation orthographique ne change presque rien ({fr(g('WER brut'))} → "
                    f"{fr(g('WER normalisé'))} %).")
        out.append(Item("table", "Normalisation : mesures", n, cap))
    a = _csv("norm_ablation_dev", root)
    if a is not None:
        r = a[a.config == "RAW + règle seule"].dropna(subset=["ΔWER"])
        best = r.loc[r["ΔWER"].idxmin()] if len(r) else None
        cap = ("Ablation règle par règle sur le <b>dev</b> (whisper-small-ft) : effet de chaque règle de normalisation "
               "sur le WER (ΔWER négatif = la règle pardonne des erreurs d'orthographe).")
        if best is not None:
            cap += (f" La règle qui compte le plus est <b>{best['règle']}</b> ({fr(best['ΔWER'])} point, "
                    f"{int(best['Δerreurs'])} erreurs) ; les autres règles orthographiques ne changent presque rien.")
        out.append(Item("table", "Ablation de la normalisation (dev)", a.round(3), cap))
    p = _img("03_phenomena", root)
    if p:
        out.append(Item("image", "Variations orthographiques", p,
                        "Phénomènes à l'origine des variantes d'un même mot dans les transcriptions : l'alef / hamza "
                        "domine (آ / أ / إ / ا), puis ya / alif maqsura."))
    return out


def dialect_items(root=RESULTS_DIR) -> list[Item]:
    out = []
    d = _csv("dialect_id_results", root)
    m = _csv("dialect_id_audio_majority", root)
    if d is not None:
        tab = pd.concat([d, m.drop(columns=["segments", "talks"], errors="ignore")], ignore_index=True) if m is not None else d
        cap = ("Identification de la région (d'après le lieu de l'événement TEDx) et du type de code-switching. "
               "Accuracy et F1 macro : plus haut = mieux. Validation croisée <b>groupée par talk</b> (un talk n'est "
               "jamais des deux côtés) ; la ligne « FUITE » montre ce que donnerait un découpage aléatoire.")
        txt = d[(d["tâche"] == "région (texte)") & d.protocole.str.startswith("GroupKFold par")]
        maj = d[d.protocole == "classe majoritaire"]
        aud = d[d["tâche"] == "région (audio)"]
        if len(txt) and len(maj):
            cap += (f"<br><b>Texte</b> : {fr(txt.iloc[0].accuracy, 3)} d'accuracy, sous la classe majoritaire "
                    f"({fr(maj.iloc[0].accuracy, 3)}) : le texte ne trahit pas la région.")
        if len(aud) and m is not None:
            cap += (f" <b>Audio</b> : {fr(aud.iloc[0].accuracy, 3)} contre {fr(m.iloc[0].accuracy, 3)} pour la classe "
                    "majoritaire sur le même échantillon et les mêmes plis. Prudence : l'audio peut reconnaître la salle "
                    "et le micro de chaque événement plutôt que l'accent.")
        out.append(Item("table", "Identification dialectale", tab, cap))
    for name, title, cap in [
        ("04_confusion_region_text", "Région par le texte : matrice de confusion",
         "Ligne = vraie région, colonne = région prédite (proportions par ligne). Une bonne prédiction remplit la "
         "diagonale ; ici les prédictions se dispersent."),
        ("04_confusion_cs_audio", "Type de code-switching par l'audio (test)",
         "Ligne = vrai type (TUN / MIXED / FOR), colonne = type prédit à partir de l'audio seul, sur le test officiel.")]:
        p = _img(name, root)
        if p:
            out.append(Item("image", title, p, cap))
    return out


def platform_items(root=RESULTS_DIR) -> list[Item]:
    t = _csv("platform_tests", root)
    if t is None:
        return []
    chrf = next((c for c in t.columns if "chrF" in c), None)
    agg = t.groupby(["mode", "cible"]).agg(exécutions=("ok", "size"), réussies=("ok", "sum"),
                                         temps_médian_s=("temps_s", "median")).reset_index()
    if chrf:
        agg = agg.merge(t[t.cible == "Anglais"].groupby("mode")[chrf].mean().round(1).rename("chrF++ indicatif (anglais)"),
                        on="mode", how="left")
    cap = (f"Tests de la plateforme : {t.seg_id.nunique()} segments du test x 3 modes x 3 langues = {len(t)} exécutions "
           f"de l'interface réelle. <b>{int(t.ok.sum())} / {len(t)}</b> sans étape en erreur ; temps médian "
           f"{fr(t.temps_s.median())} s par exécution. Le chrF++ (plus haut = mieux) n'est qu'indicatif : 20 segments.")
    return [Item("table", "Tests de la plateforme", agg.round(2), cap)]


SECTIONS = [("Reconnaissance de la parole", asr_items), ("Traduction", ast_items), ("Normalisation", norm_items),
            ("Identification dialectale", dialect_items), ("Plateforme", platform_items)]
