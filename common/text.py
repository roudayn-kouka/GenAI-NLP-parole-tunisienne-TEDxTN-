"""
Traitement du texte tunisien : détection de script, code-switching, CMI,
normalisation orthographique et comparaison inter-scripts.

Ce module ne dépend que de la bibliothèque standard : il est utilisé par
les notebooks (analyse, évaluation, normalisation) et par l'interface.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Iterable

# --------------------------------------------------------------------------
# Classes de caractères
# --------------------------------------------------------------------------
_AR_RANGES = r"؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
AR_CHAR = re.compile(f"[{_AR_RANGES}]")
AR_LETTER = re.compile(r"[ء-غف-يٱ-ۓڤڨگ]")
LAT_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
DIGIT = re.compile(r"[0-9٠-٩۰-۹]")

TASHKEEL = re.compile(r"[ً-ْٰٓ-ٟۖ-ۭ]")
TATWEEL = "ـ"

# Hésitations / marques d'oralité fréquentes dans TEDxTN (cf. Table 4 du papier : "آ" = hesitation).
# Formes d'ORIGINE (avant normalisation de l'alef) : « أم » n'en fait pas partie (mot), « آم » / « إم » oui.
HESITATIONS = {"آ", "أ", "إ", "آم", "إم", "أه", "اه", "آه", "إه",
               "ا", "اا", "آآ", "ااا", "ام", "امم", "ممم", "مم", "هم", "همم",
               "euh", "eh", "hmm", "hm", "mmm", "uh", "um", "ah", "bah", "ben"}

# Annotations éventuelles entre crochets/chevrons : [rire], <bruit>, (inaudible)
BRACKETED = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")

# Ponctuation (latine + arabe)
PUNCT = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~«»“”‘’…،؛؟٪٫٬﴾﴿–—·]")

_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


# --------------------------------------------------------------------------
# Script et code-switching
# --------------------------------------------------------------------------
def token_script(tok: str) -> str:
    """Retourne le script d'un token :
    'ar'    : uniquement lettres arabes (tunisien écrit en arabe)
    'lat'   : uniquement lettres latines (mot français/anglais)
    'mixed' : les deux dans le même token (code-switching intra-mot, ex. 'الfumoir')
    'num'   : chiffres
    'other' : ponctuation, symboles, vide
    """
    has_ar = bool(AR_LETTER.search(tok))
    has_lat = bool(LAT_LETTER.search(tok))
    if has_ar and has_lat:
        return "mixed"
    if has_ar:
        return "ar"
    if has_lat:
        return "lat"
    if DIGIT.search(tok):
        return "num"
    return "other"


def tag_tokens(text: str) -> list[tuple[str, str]]:
    """Découpe en tokens et associe à chacun son script. Les hésitations sont taguées 'hes'."""
    out = []
    for tok in str(text).split():
        clean = PUNCT.sub("", tok)
        if not clean:
            continue
        tag = "hes" if clean.lower() in HESITATIONS else token_script(clean)
        out.append((tok, tag))
    return out


def segment_type(text: str) -> str:
    """Type de segment au sens de la Table 9 du papier :
    TUN   : uniquement du tunisien (script arabe)
    FOR   : uniquement des mots étrangers (script latin)
    MIXED : code-switching (au moins un mot de chaque, ou un mot mixte)
    """
    tags = [t for _, t in tag_tokens(text) if t in {"ar", "lat", "mixed"}]
    if not tags:
        return "TUN"
    s = set(tags)
    if s == {"ar"}:
        return "TUN"
    if s == {"lat"}:
        return "FOR"
    return "MIXED"


def cmi(text: str) -> float:
    """Code-Mixing Index (Das & Gambäck, 2014), en pourcentage.
    CMI = 100 * (sum(w_i) - max(w_i)) / (n - u)
    w_i : nombre de mots de la langue i ; n : nombre total de tokens ;
    u : tokens indépendants de la langue (chiffres, hésitations, ponctuation).
    Les tokens 'mixed' sont comptés comme étrangers (leur radical est étranger)."""
    tags = [t for _, t in tag_tokens(text)]
    n = len(tags)
    u = sum(t in {"num", "other", "hes"} for t in tags)
    if n - u <= 0:
        return 0.0
    counts = Counter("ar" if t == "ar" else "for" for t in tags if t in {"ar", "lat", "mixed"})
    if not counts:
        return 0.0
    total = sum(counts.values())
    return 100.0 * (total - max(counts.values())) / (n - u)


def corpus_cmi(texts: Iterable[str]) -> float:
    """CMI au niveau du corpus = moyenne des CMI des énoncés."""
    vals = [cmi(t) for t in texts]
    return sum(vals) / len(vals) if vals else 0.0


def code_switch_stats(texts: Iterable[str]) -> dict:
    """Statistiques globales de code-switching (comparables à la Table 3 du papier)."""
    seg_types = Counter()
    tok_types = Counter()
    n = 0
    for t in texts:
        n += 1
        seg_types[segment_type(t)] += 1
        tok_types.update(tag for _, tag in tag_tokens(t))
    words = sum(tok_types[k] for k in ("ar", "lat", "mixed"))
    return {
        "segments": n,
        "seg_TUN": seg_types["TUN"], "seg_MIXED": seg_types["MIXED"], "seg_FOR": seg_types["FOR"],
        "pct_segments_with_foreign": 100 * (seg_types["MIXED"] + seg_types["FOR"]) / max(n, 1),
        "words_ar": tok_types["ar"], "words_foreign": tok_types["lat"], "words_intra_cs": tok_types["mixed"],
        "pct_words_foreign": 100 * tok_types["lat"] / max(words, 1),
        "pct_words_intra_cs": 100 * tok_types["mixed"] / max(words, 1),
        "hesitations": tok_types["hes"],
    }


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
@dataclass
class NormConfig:
    """Conventions de normalisation (documentées dans le notebook 03).

    Profil par défaut (= EVAL_NORM) : normalisation orthographique seule, utilisée pour le WER/CER normalisé.
    Elle neutralise les variations qui ne changent pas le mot prononcé, sans retirer de mots ni en couper :
    - remove_hesitations : variante EVAL_NORM_NOHES, rapportée à part (« WER sans hésitations ») ;
    - split_mixed_tokens : convention d'écriture (notebook 03, WRITING_NORM), pas d'évaluation, car la règle est
      asymétrique (les références écrivent les emprunts en arabe, seules les hypothèses ont des mots hybrides).
    """
    remove_brackets: bool = True         # [rire], <bruit>
    remove_diacritics: bool = True       # harakat + tatweel
    normalize_alef: bool = True          # أ إ آ ٱ -> ا
    normalize_ya: bool = True            # ى -> ي
    normalize_ta_marbuta: bool = False   # ة -> ه (désactivé : distinction souvent utile)
    normalize_hamza: bool = False        # ؤ -> و, ئ -> ي
    normalize_gaf: bool = True           # گ ڭ ݣ -> ڨ (le /g/ tunisien a plusieurs graphies)
    normalize_veh: bool = False          # ڤ -> ف
    arabic_digits: bool = True           # ٣ -> 3
    remove_punctuation: bool = True
    lowercase_latin: bool = True
    strip_latin_accents: bool = False    # é -> e (option : erreurs d'accents fréquentes)
    collapse_elongation: bool = True     # برشاااا -> برشا, cooool -> cool
    split_mixed_tokens: bool = False     # الfumoir -> ال fumoir (frontière de script) : convention d'écriture
    remove_hesitations: bool = False     # آ, euh, ممم (forme d'origine du token) : EVAL_NORM_NOHES
    loanword_lexicon: dict | None = None  # {"سيركيلاسيون": "circulation"} (option, cf. notebook 03)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["loanword_lexicon"] = len(self.loanword_lexicon or {})
        return d


EVAL_NORM = NormConfig()                                   # WER normalisé : orthographe seule
EVAL_NORM_NOHES = NormConfig(remove_hesitations=True)      # WER sans hésitations : EVAL_NORM + hésitations retirées
WRITING_NORM = NormConfig(split_mixed_tokens=True)         # convention d'écriture (notebook 03, interface)
RAW_NORM = NormConfig(  # proche du protocole du papier : ponctuation retirée, casse latine ignorée
    remove_diacritics=False, normalize_alef=False, normalize_ya=False, normalize_gaf=False,
    arabic_digits=False, collapse_elongation=False, split_mixed_tokens=False, remove_hesitations=False,
)


def is_hesitation(tok: str) -> bool:
    """Hésitation repérée sur la forme d'origine du token (avant normalisation de l'alef) : ponctuation,
    diacritiques et tatweel ignorés, casse latine ignorée, allongements (3+ lettres répétées) ramenés à 2.
    Ainsi « آم » (hésitation) est retiré mais « أم » (mot) est gardé, même si l'alef est normalisé ensuite."""
    t = TASHKEEL.sub("", PUNCT.sub("", tok)).replace(TATWEEL, "").lower()
    return bool(t) and (t in HESITATIONS or re.sub(r"(.)\1{2,}", r"\1\1", t) in HESITATIONS)

_SCRIPT_BOUNDARY = re.compile(
    r"(?<=[ء-يٱ-ۓ])(?=[A-Za-zÀ-ɏ])|(?<=[A-Za-zÀ-ɏ])(?=[ء-يٱ-ۓ])"
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_text(text: str, cfg: NormConfig = EVAL_NORM, return_trace: bool = False):
    """Normalise une transcription tunisienne selon `cfg`.

    Si `return_trace=True`, renvoie aussi la liste des règles appliquées avec
    l'état du texte après chaque règle (utilisé par l'interface pour montrer
    la normalisation étape par étape)."""
    trace = []

    def step(name, fn, s):
        new = fn(s)
        if return_trace and new != s:
            trace.append({"rule": name, "before": s, "after": new})
        return new

    s = unicodedata.normalize("NFC", str(text))
    if cfg.remove_brackets:
        s = step("annotations entre crochets", lambda x: BRACKETED.sub(" ", x), s)
    if cfg.remove_hesitations:   # sur la forme d'origine, avant l'alef : « أم » reste, « آم » part
        s = step("hésitations", lambda x: " ".join(w for w in x.split() if not is_hesitation(w)), s)
    if cfg.remove_diacritics:
        s = step("diacritiques et tatweel", lambda x: TASHKEEL.sub("", x).replace(TATWEEL, ""), s)
    if cfg.normalize_alef:
        s = step("alef (أ إ آ ٱ → ا)", lambda x: re.sub("[أإآٱ]", "ا", x), s)
    if cfg.normalize_ya:
        s = step("alif maqsura (ى → ي)", lambda x: x.replace("ى", "ي"), s)
    if cfg.normalize_ta_marbuta:
        s = step("ta marbuta (ة → ه)", lambda x: x.replace("ة", "ه"), s)
    if cfg.normalize_hamza:
        s = step("hamza (ؤ ئ)", lambda x: x.replace("ؤ", "و").replace("ئ", "ي"), s)
    if cfg.normalize_gaf:
        s = step("/g/ tunisien (گ ڭ → ڨ)", lambda x: re.sub("[گڭݣ]", "ڨ", x), s)
    if cfg.normalize_veh:
        s = step("veh (ڤ → ف)", lambda x: x.replace("ڤ", "ف"), s)
    if cfg.arabic_digits:
        s = step("chiffres arabes-indiens", lambda x: x.translate(_AR_DIGITS), s)
    if cfg.remove_punctuation:
        s = step("ponctuation", lambda x: PUNCT.sub(" ", x), s)
    if cfg.lowercase_latin:
        s = step("minuscules (latin)", lambda x: x.lower(), s)
    if cfg.strip_latin_accents:
        s = step("accents latins", lambda x: "".join(
            _strip_accents(c) if LAT_LETTER.match(c) else c for c in x), s)
    if cfg.collapse_elongation:
        def _collapse(x):
            x = re.sub(r"([ء-ي])\1{2,}", r"\1", x)      # arabe : 3+ -> 1
            return re.sub(r"([a-zà-ÿ])\1{2,}", r"\1\1", x)        # latin : 3+ -> 2
        s = step("allongements expressifs", _collapse, s)
    if cfg.split_mixed_tokens:
        s = step("frontière de script (الfumoir → ال fumoir)", lambda x: _SCRIPT_BOUNDARY.sub(" ", x), s)
    if cfg.loanword_lexicon:
        lex = cfg.loanword_lexicon
        s = step("lexique d'emprunts (script unifié)",
                 lambda x: " ".join(lex.get(w, w) for w in x.split()), s)
    s = re.sub(r"\s+", " ", s).strip()
    return (s, trace) if return_trace else s


# --------------------------------------------------------------------------
# Comparaison inter-scripts (erreurs de script, cf. Table 8 du papier)
# --------------------------------------------------------------------------
_AR2LAT = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ٱ": "a", "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "d", "ر": "r", "ز": "z", "س": "s", "ش": "ch", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "d", "ع": "", "غ": "gh", "ف": "f", "ق": "k", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "ou", "ي": "i", "ى": "a", "ة": "a", "ء": "", "ؤ": "ou", "ئ": "i", "ڤ": "v",
    "ڨ": "g", "گ": "g", "پ": "p", "چ": "tch",
}


def arabic_to_latin(word: str) -> str:
    """Translittération grossière arabe -> latin (sert uniquement à comparer des sons)."""
    return "".join(_AR2LAT.get(c, "") for c in word)


def phonetic_skeleton(word: str) -> str:
    """Squelette consonantique indépendant du script.
    'circulation' et 'سيركيلاسيون' donnent tous deux 'srklsn'."""
    w = word.lower()
    if AR_LETTER.search(w):
        w = re.sub(r"^ال(?=..)", "", w)  # article défini
        w = arabic_to_latin(w)
    w = _strip_accents(w)
    w = w.replace("tion", "sion").replace("ph", "f").replace("qu", "k").replace("x", "ks")
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k").replace("q", "k")
    w = re.sub(r"[aeiouyw'h]", "", w)
    w = re.sub(r"(.)\1+", r"\1", w)
    return w


def is_script_variant(ref_word: str, hyp_word: str, threshold: float = 0.75) -> bool:
    """Vrai si deux mots écrits dans des scripts différents se prononcent probablement pareil.
    Utilisé pour isoler les « fausses erreurs » de script dans l'analyse ASR."""
    s1, s2 = token_script(ref_word), token_script(hyp_word)
    if s1 == s2 or "other" in (s1, s2) or "num" in (s1, s2):
        return False
    a, b = phonetic_skeleton(ref_word), phonetic_skeleton(hyp_word)
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def is_spelling_variant(ref_word: str, hyp_word: str, cfg: NormConfig = EVAL_NORM) -> bool:
    """Vrai si deux mots du même script deviennent identiques après normalisation."""
    return ref_word != hyp_word and normalize_text(ref_word, cfg) == normalize_text(hyp_word, cfg) != ""


# --------------------------------------------------------------------------
# Rendu HTML (interface et notebooks)
# --------------------------------------------------------------------------
TAG_COLORS = {
    "ar": ("#0f766e", "Tunisien (arabe)"),
    "lat": ("#b45309", "Étranger (latin)"),
    "mixed": ("#7c3aed", "Intra-mot"),
    "hes": ("#64748b", "Hésitation"),
    "num": ("#2563eb", "Nombre"),
    "other": ("#64748b", "Autre"),
}


def code_switch_html(text: str) -> str:
    """Colore chaque mot selon son script (utilisé dans l'interface)."""
    import html as _html
    spans = []
    for tok, tag in tag_tokens(text):
        color, label = TAG_COLORS.get(tag, TAG_COLORS["other"])
        spans.append(
            f'<span class="cs-tok cs-{tag}" title="{label}" '
            f'style="--c:{color}">{_html.escape(tok)}</span>'
        )
    return '<div class="cs-line" dir="rtl">' + " ".join(spans) + "</div>"
