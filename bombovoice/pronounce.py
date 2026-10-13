"""Turn written Patois into text an English voice will *say* (not spell out).

Patois spellings like "seh", "nuh", "cyaan" or "bwoy" read nicely, but text-to-speech
engines don't know them and often spell them letter by letter ("S-E-H"). ALL-CAPS
shouting ("YA MON!") gets read like an acronym. This module rewrites the display text
into English-looking spellings that produce the right sounds, without changing what is
shown on screen.
"""

from __future__ import annotations

import re

# patois spelling -> spelling an English TTS voice pronounces correctly
SAY_AS: dict[str, str] = {
    # pronouns & little words
    "mi": "me", "miself": "meeself", "yuh": "ya", "yuhself": "yaself", "unu": "oonoo",
    "im": "him", "imself": "himself", "shi": "she", "har": "har", "wi": "wee", "wiself": "weeself",
    "dem": "dem", "demself": "demself", "di": "dee", "dis": "dis", "dat": "dat", "dese": "deese",
    "deh": "dey", "dere": "dare", "ya": "yah", "weh": "way", "wah": "wah", "wen": "wen",
    "fi": "fee", "pon": "pon", "wid": "wid", "widout": "widout", "becau": "becaw", "bout": "bout",
    "an'": "an", "an": "an", "jus'": "jus", "nuh": "noh", "neva": "nevva", "eva": "evva",
    "suh": "so", "seh": "say", "nah": "nah", "cyaan": "kahn", "caan": "kahn", "kyaan": "kahn",
    "haffi": "haffee", "waan": "wahn", "gwaan": "gwahn", "gwine": "gwine", "fa": "fah",
    "smaddy": "smaddy", "anybaddy": "anybaddy", "nutt'n": "nuttin", "nuttn": "nuttin", "sinting": "sinting",
    "anyting": "anyting", "everyting": "everyting", "ting": "ting", "tings": "tings", "tink": "tink",
    # people
    "bredda": "bredda", "bredren": "bredren", "bredrin": "bredrin", "sistren": "sistren", "fren": "fren",
    "bwoy": "buh-woy", "gyal": "gal", "ooman": "ooman", "pickney": "pickny", "yute": "yoot",
    "mumma": "mumma", "puppa": "puppa", "eediat": "eedee-at", "bossman": "boss man", "mon": "mahn",
    "wifey": "wifey", "teacha": "teacha", "madda": "madda", "fada": "fahda",
    # verbs
    "nyam": "nee-yahm", "nyamin": "nee-yahmin", "mek": "mek", "tek": "tek", "gi": "ghee", "lef": "leff",
    "siddung": "sid dung", "tief": "teef", "bawl": "bawl", "skank": "skank", "chat": "chat",
    "overstand": "overstand", "overstan'": "overstand", "cyaan stand": "kahn stand",
    # places, things, time
    "yaad": "yard", "riddim": "riddim", "tune": "tune", "dollaz": "dollaz", "dawg": "dawg", "puss": "puss",
    "yeye": "yeh yeh", "dinna": "dinna", "wata": "wahta", "computa": "computa",
    "mawnin": "mawnin", "tinite": "tuh-nite", "tiday": "tuh-day", "tumaro": "tuh-mah-row",
    "yessideh": "yes-suh-day", "lata": "lata", "bifo": "bih-foh", "afta": "afta", "ova": "ova", "unda": "unda",
    "likkle": "lickle", "islan'": "island",
    # vibes & exclamations
    "irie": "eye-ree", "criss": "criss", "propa": "propa", "nuff": "nuff", "bare": "bare", "fass": "fass",
    "tyad": "tyard", "fraid": "fraid", "vex": "vex", "dread": "dread", "massive": "massive",
    "fenky-fenky": "fenky fenky", "rahtid": "rah-tid", "lawd": "lawd", "jah": "jah", "yow": "yow",
    "zeet": "zeet", "zimmi": "zimmee", "seen": "seen", "cho": "choh", "yah": "yah", "mussi": "mussee",
    "bomboclaat": "bombo-claht", "bumboclaat": "bumbo-claht", "raasclaat": "rahs-claht",
    "bloodclaat": "blood-claht", "claat": "claht", "wheeeeel": "wheeel", "inna": "inna",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z']|[A-Za-z]")


def _say_word(word: str) -> str:
    low = word.lower()
    if low in SAY_AS:
        return SAY_AS[low]
    stripped = low.strip("'")
    if stripped in SAY_AS:
        return SAY_AS[stripped]
    # "nyamin'" / "runnin'" -> "nyamin" / "runnin"; "'ave" -> "ave"
    stripped = stripped.replace("'", "")
    if not stripped:
        return ""
    # unknown words ending in "eh"/"uh" ("seh", "deh") get spelled out by many voices
    stripped = re.sub(r"eh$", "ay", stripped)
    stripped = re.sub(r"(?<=[a-z]{2})uh$", "a", stripped)
    # a "word" with no vowel would be spelled out letter by letter
    if not re.search(r"[aeiouy]", stripped):
        return stripped if len(stripped) > 3 else ""
    return stripped


def speakable(text: str) -> str:
    """Display Patois -> text for the TTS engine (lowercase-friendly, no acronym traps)."""
    # multi-word entries first
    text = re.sub(r"(?i)\bcyaan stand\b", "kahn stand", text)
    out = _WORD.sub(lambda m: _say_word(m.group(0)), text)
    out = out.replace("?!", "?").replace("!!", "!")
    out = re.sub(r"\s+", " ", out).strip()
    # sentence-case so nothing is read as an acronym
    out = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out
