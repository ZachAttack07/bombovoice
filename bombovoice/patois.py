"""Offline English -> Jamaican Patois translator.

This is the no-internet, no-API-key brain of Bombovoice. It rewrites English
into phonetically respelled Patois ("the" -> "di", "think" -> "tink") so that
*any* English text-to-speech voice ends up pronouncing it with a Jamaican
flavour, and it splits the result into prosody segments (emphasis + pitch
contour) that the TTS renderer turns into that sing-song yaad delivery.

Intensity levels:
    0  Tourist      - light word swaps, still mostly English
    1  Yaadie       - proper Patois grammar and pronunciation
    2  Full Patois  - plural "dem", h-dropping, more tags and ad-libs
    3  MAXIMUM      - everything turned up, h-insertion, ad-libs everywhere
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

INTENSITY_NAMES = {0: "Tourist", 1: "Yaadie", 2: "Full Patois", 3: "MAXIMUM BOMBO"}

CONTOURS = ("flat", "rise", "fall", "rise_fall", "high", "low")


@dataclass
class Segment:
    """One chunk of speech with its own delivery."""

    text: str
    emphasis: int = 0  # 0 = normal ... 3 = SHOUT IT
    contour: str = "flat"  # one of CONTOURS
    pause_after_ms: int = 120


@dataclass
class Translation:
    original: str
    patois: str
    segments: list[Segment] = field(default_factory=list)
    source: str = "rules"  # "rules" or "ai"


# ---------------------------------------------------------------------------
# Phrase-level rewrites (run on lowercased text, longest patterns first)
# ---------------------------------------------------------------------------

_PHRASES: list[tuple[str, str]] = [
    (r"what'?s up|what is up|what'?s going on|what is going on|how'?s it going|how are you( doing)?|how you doing", "wah gwaan"),
    (r"what'?s happening|what is happening", "wah a gwaan"),
    (r"oh my (god|gosh|goodness)|omg", "lawd have mercy"),
    (r"thank you( very much| so much)?|thanks( a lot)?", "tank yuh, bless up"),
    (r"good ?bye|bye bye|see you later|see ya( later)?|talk to you later|later guys", "likkle more"),
    (r"good morning", "mawnin"),
    (r"good night", "good night, sleep irie"),
    (r"never mind|forget about it|forget it", "nuh worry yuhself"),
    (r"don'?t worry( about it)?", "nuh worry yuhself"),
    (r"calm down|chill out|relax", "cool yuhself"),
    (r"no problem|no worries|you'?re welcome", "no problem mon"),
    (r"i don'?t know|dunno", "mi nuh know"),
    (r"do you understand|you know what i mean|you get me|understand\?", "yuh overstand?"),
    (r"look at", "look pon"),
    (r"a lot of|lots of|plenty of|loads of|tons of", "nuff"),
    (r"a little bit|a bit", "a likkle"),
    (r"right now", "right now so"),
    (r"come on", "cho"),
    (r"let'?s go", "mek wi go"),
    (r"let me", "mek mi"),
    (r"let us", "mek wi"),
    (r"have to|has to|got to|gotta|need to|needs to", "haffi"),
    (r"want to|wanna|wants to", "waan"),
    (r"going to|gonna", "a go"),
    (r"used to", "did useta"),
    (r"to the", "a di"),
    (r"there'?s|there is|there are", "dere a"),
    (r"(can'?t|cannot|can not)", "cyaan"),
    (r"(won'?t|will not)", "nah go"),
    (r"(didn'?t|did not|wasn'?t|was not|weren'?t|were not|haven'?t|have not|hasn'?t|has not)", "neva"),
    (r"(don'?t|doesn'?t|do not|does not|isn'?t|is not|aren'?t|are not|ain'?t|am not)", "nuh"),
    (r"do you", "yuh"),
    (r"are you", "yuh"),
    (r"is it", "a it"),
    (r"all right|alright|okay|ok\b", "irie"),
    (r"for real|seriously|no cap", "fi real"),
    (r"right\?$", "seen?"),
]

_COMPILED_PHRASES = [(re.compile(r"\b(?:" + p + r")\b" if not p.endswith("$") else r"\b(?:" + p + r")"), r) for p, r in _PHRASES]

# subject pronoun -> patois subject
_SUBJECTS = {"i": "mi", "you": "yuh", "he": "im", "she": "shi", "we": "wi", "they": "dem", "it": "it", "that": "dat", "this": "dis"}

# subject + copula ("i'm", "you are", "he's"...) followed by the next word
_COPULA_RE = re.compile(
    r"\b(i|you|he|she|we|they|it|that|this)(?:\s+(?:am|are|is|was|were)|'m|'re|'s)\s+([a-z']+)"
)

# ---------------------------------------------------------------------------
# Word-level dictionary
# ---------------------------------------------------------------------------

_WORDS: dict[str, str | tuple[str, ...]] = {
    # pronouns & determiners
    "i": "mi", "me": "mi", "my": "mi", "mine": "fi mi", "myself": "miself",
    "you": "yuh", "your": "yuh", "yours": "fi yuh", "yourself": "yuhself", "ya": "yuh", "u": "yuh",
    "he": "im", "him": "im", "his": "im", "himself": "imself",
    "she": "shi", "her": "har", "herself": "harself",
    "we": "wi", "us": "wi", "our": "wi", "ourselves": "wiself",
    "they": "dem", "them": "dem", "their": "dem", "theirs": "fi dem", "themselves": "demself",
    "the": "di", "this": "dis", "that": "dat", "these": "dese", "those": "dem deh",
    "there": "deh", "here": "ya", "where": "weh", "what": "wah", "when": "wen", "why": "wah mek",
    "with": "wid", "without": "widout", "about": "bout", "around": "round", "because": "becau", "cause": "cause",
    "for": "fi", "on": "pon", "upon": "pon", "of": "a", "to": "fi", "and": "an'", "just": "jus'",
    "something": "sinting", "nothing": "nutt'n", "anything": "anyting", "everything": "everyting",
    "someone": "smaddy", "somebody": "smaddy", "everyone": "everybody", "anybody": "anybaddy",
    # people
    "friend": ("bredren", "fren", "bredda"), "friends": "bredren dem", "buddy": "bredda",
    "bro": ("bredda", "bredrin"), "brother": "bredda", "dude": ("bredda", "mi yute"), "man": "mon", "guy": "bwoy",
    "guys": "bredren dem", "sister": "sistren", "sis": "sistren", "girl": "gyal", "girls": "gyal dem",
    "woman": "ooman", "women": "ooman dem", "boy": "bwoy", "boys": "bwoy dem",
    "kid": "pickney", "kids": "pickney dem", "child": "pickney", "children": "pickney dem", "baby": "baby",
    "mother": "madda", "mom": "mumma", "mum": "mumma", "father": "fada", "dad": "puppa", "grandma": "granny",
    "people": "people dem", "everybody": "everybody", "boss": "bossman", "teacher": "teacha",
    "girlfriend": "gyal", "boyfriend": "man", "wife": "wifey", "husband": "mi man",
    "fool": "fool", "idiot": "eediat", "stupid": "eediat", "dumb": "eediat",
    # verbs (Patois uses the base form for the past tense)
    "say": "seh", "says": "seh", "said": "seh", "eat": "nyam", "eats": "nyam", "ate": "nyam", "eating": "nyamin",
    "give": "gi", "gives": "gi", "gave": "gi", "talk": "chat", "talks": "chat", "talked": "chat", "talking": "a chat",
    "make": "mek", "makes": "mek", "made": "mek", "take": "tek", "takes": "tek", "took": "tek",
    "think": "tink", "thinks": "tink", "thought": "tink", "thinking": "a tink",
    "leave": "lef", "left": "lef", "went": "go", "came": "come", "saw": "see", "seen": "see", "told": "tell",
    "got": "get", "gotten": "get", "bought": "buy", "brought": "bring", "sat": "siddung", "sit": "siddung",
    "understand": "overstand", "understood": "overstand", "know": "know", "knew": "did know",
    "want": "waan", "wants": "waan", "wanted": "did waan", "like": "like", "love": "love", "hate": "cyaan stand",
    "look": "look", "watch": "watch", "listen": "lissen", "wait": "hol' on", "stop": "stop", "run": "run",
    "go": "go", "going": "a go", "come": "come", "coming": "a come", "dance": "skank", "dancing": "a skank",
    "relax": "cool out", "sleep": "sleep", "sleeping": "a sleep", "work": "work", "working": "a work",
    "laugh": "laugh", "laughing": "a dead wid laugh", "cry": "bawl", "crying": "a bawl", "shout": "bawl out",
    "yell": "bawl out", "steal": "tief", "stole": "tief", "lie": "tell lie", "lying": "a tell lie",
    "fight": "fight", "kill": "kill", "die": "dead", "died": "dead", "dead": "dead",
    "should": "shoulda", "could": "coulda", "would": "woulda", "must": "mus'", "might": "might",
    "can": "can", "will": "wi", "was": "did", "were": "did",
    "please": "beg yuh", "sorry": "mi sorry",
    # things & places
    "house": "yaad", "home": "yaad", "yard": "yaad", "food": "food", "dinner": "dinna", "water": "wata",
    "money": ("money", "dollaz"), "cash": "dollaz", "car": "ride", "phone": "phone", "dog": "dawg",
    "cat": "puss", "thing": "ting", "things": "tings", "stuff": "tings", "party": "dance",
    "music": "riddim", "song": "tune", "songs": "tune dem", "game": "game", "computer": "computa",
    "street": "road", "city": "town", "country": "country", "island": "islan'", "beach": "beach",
    "head": "head", "eyes": "yeye dem", "eye": "yeye", "egg": "egg", "ice": "ice",
    "problem": "problem", "trouble": "trouble", "fun": "nuff fun", "time": "time",
    "morning": "mawnin", "tonight": "tinite", "today": "tiday", "tomorrow": "tumaro", "yesterday": "yessideh",
    "later": "lata", "now": "now", "always": "all de while", "never": "neva", "ever": "eva",
    "again": "again", "before": "bifo", "after": "afta", "over": "ova", "under": "unda",
    "little": "likkle", "small": "likkle", "tiny": "likkle likkle", "big": "big", "huge": "big big",
    "very": ("real", "bare", "propa"), "really": ("real", "fi real"), "so": "suh", "too": "too",
    "many": "nuff", "much": "nuff", "lots": "nuff", "plenty": "nuff", "more": "more",
    # vibes
    "good": ("irie", "good", "criss"), "great": ("wicked", "irie", "massive"), "nice": ("irie", "nice", "criss"),
    "awesome": ("wicked", "massive", "bad"), "amazing": ("wicked", "massive"), "cool": ("irie", "criss"),
    "excellent": "wicked", "perfect": "criss", "beautiful": "pretty", "pretty": "pretty", "cute": "pretty",
    "bad": "bad", "terrible": "dread", "awful": "dread", "horrible": "dread", "scary": "dread",
    "crazy": "mad", "insane": "mad", "angry": "vex", "mad": "vex", "annoyed": "vex", "upset": "vex",
    "tired": "tyad", "hungry": "hungry", "happy": "irie", "sad": "sad", "scared": "fraid", "afraid": "fraid",
    "strong": "strong", "weak": "fenky-fenky", "lazy": "lazy", "fast": "fass", "quick": "quick", "slow": "slow",
    "yes": ("yes", "ya mon", "yeah mon"), "yeah": ("yeah mon", "yes", "ya mon"), "yep": "ya mon",
    "no": ("no", "nuh"), "nope": "nuh mon", "hello": "wah gwaan", "hi": "wah gwaan", "hey": "yow",
    "respect": "respect", "true": "true", "wow": "bwoy", "damn": "rahtid", "dang": "rahtid", "god": "Jah",
    "lord": "lawd", "okay": "irie", "ok": "irie", "lol": "a dead wid laugh",
}

# function words whose initial "th" is voiced -> "d" (everything else -> "t")
_VOICED_TH = re.compile(r"^th(?:e|is|at|ese|ose|ey|em|en|an|ere|eir|us|ough|ee|y|erefore|ence)(?:'.*|selves)?$")

_ING_STEMS = {
    "making": "mek", "taking": "tek", "having": "have", "giving": "gi", "living": "live", "driving": "drive",
    "writing": "write", "using": "use", "leaving": "lef", "loving": "love", "hoping": "hope", "saying": "seh",
    "doing": "do", "trying": "try", "playing": "play", "getting": "get", "sitting": "siddung", "running": "run",
    "waiting": "wait", "looking": "look", "telling": "tell", "moving": "move", "smoking": "smoke",
    "dancing": "skank", "chilling": "cool out", "eating": "nyam", "talking": "chat", "thinking": "tink",
    "going": "go", "coming": "come", "staying": "stay", "losing": "lose", "winning": "win", "shooting": "shoot",
    "cooking": "cook", "singing": "sing", "joking": "joke", "lying": "tell lie", "praying": "pray",
}

_INTENSIFIERS = {"nuff", "real", "bare", "propa", "wicked", "massive", "big", "irie", "criss", "dread", "mad",
                 "vex", "rahtid", "bwoy", "lawd", "jah", "fi", "haffi", "cyaan", "neva", "suh", "likkle"}

_OPENERS = {
    0: ["Yeah mon,", "Mi seh,"],
    1: ["Bwoy,", "Yow,", "Mi seh,", "Lissen,", "Ya mon,"],
    2: ["Bwoy,", "Yow!", "Lawd,", "Mi seh,", "Hear mi now,", "Bredda,", "Rahtid,"],
    3: ["BWOY!", "YOW YOW YOW!", "Lawd Jesus,", "Hear mi now!", "Rahtid!", "Wheeeeel!", "Jah know,"],
}
_SPICY_OPENERS = ["Bomboclaat!", "Raasclaat!", "Bloodclaat!", "Bumboclaat!"]

_TAGS = {
    0: [", ya mon", ""],
    1: [", ya mon", ", seen?", ", mi bredda", ", yuh zeet?", ", fi real"],
    2: [", seen?", ", ya mon", ", star", ", yuh zimmi?", ", mi bredren", ", nuh true?", ", fi real doh"],
    3: [", SEEN?!", ", YA MON!", ", star!", ", Jah know!", ", nuh true?!", ", big tune!", ", yuh zimmi?!"],
}


def _pick(value: str | tuple[str, ...], rng: random.Random) -> str:
    return rng.choice(value) if isinstance(value, tuple) else value


def _ing_stem(word: str) -> str:
    if word in _ING_STEMS:
        return _ING_STEMS[word]
    stem = word[:-3]
    if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiouls":
        stem = stem[:-1]  # running -> run
    return stem


def _phonetic(word: str, intensity: int, rng: random.Random) -> str:
    """Respell an English word the way a yaadie would say it."""
    w = word
    if len(w) <= 2:
        return w
    # th- at the start
    if w.startswith("th"):
        w = ("d" if _VOICED_TH.match(w) else "t") + w[2:]
    # th in the middle / end: brother -> bredda style, mouth -> mout
    if "th" in w[1:]:
        if w.endswith("th"):
            w = w[:-2] + "t"
        w = w.replace("th", "d")
    # -ttle / -tle -> kkle (bottle -> bokkle)
    w = re.sub(r"t+le$", "kkle", w)
    # -ing -> -in'
    if w.endswith("ing") and len(w) > 4:
        w = w[:-3] + "in'"
    # -er -> -a (never -> neva, better -> betta)
    elif w.endswith("er") and len(w) > 3:
        w = w[:-2] + "a"
    elif w.endswith("ers") and len(w) > 4:
        w = w[:-3] + "as"
    # consonant cluster reduction at the end: hand -> han', best -> bes'
    if intensity >= 1 and len(w) > 3:
        if re.search(r"[aeiou]nd$", w):
            w = w[:-1] + "'"
        elif re.search(r"[aeiou]st$", w):
            w = w[:-1] + "'"
    # h-dropping / h-insertion for the full experience
    if intensity >= 2 and w.startswith("h") and len(w) > 3 and rng.random() < 0.35:
        w = "'" + w[1:]
    elif intensity >= 3 and w[0] in "aeiou" and len(w) > 2 and rng.random() < 0.15:
        w = "h" + w
    return w


def _copula(match: re.Match) -> str:
    subject, nxt = match.group(1), match.group(2)
    full = match.group(0)
    past = bool(re.search(r"\s(was|were)\s", full))
    subj = _SUBJECTS[subject]
    if nxt.endswith("ing") and len(nxt) > 4:
        aux = "did a" if past else "a"
        return f"{subj} {aux} {_ing_stem(nxt)}"
    if nxt in ("a", "an", "the", "my", "your", "his", "her", "our", "their"):
        return f"{subj} {'did ' if past else ''}a {nxt}" if nxt not in ("a", "an") else f"{subj} {'did ' if past else ''}a"
    if nxt in ("not", "never"):
        return f"{subj} {'neva' if past else 'nuh'}"
    # adjectives: Patois drops the copula entirely ("mi happy")
    return f"{subj} {'did ' if past else ''}{nxt}"


_TOKEN_RE = re.compile(r"[A-Za-z']+|[^A-Za-z']+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*")


class PatoisTranslator:
    """Rule-based translator. Deterministic when given a seed."""

    def __init__(self, intensity: int = 2, spicy: bool = False, seed: int | None = None):
        self.intensity = max(0, min(3, int(intensity)))
        self.spicy = spicy
        self.rng = random.Random(seed)

    # -- public -------------------------------------------------------------
    def translate(self, text: str) -> Translation:
        text = text.strip()
        if not text:
            return Translation(original=text, patois="", segments=[])
        sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
        out_sentences = [self._sentence(s, i) for i, s in enumerate(sentences)]
        patois = " ".join(s for s in out_sentences if s)
        return Translation(original=text, patois=patois, segments=self.segment(patois), source="rules")

    def translate_text(self, text: str) -> str:
        return self.translate(text).patois

    # -- internals ------------------------------------------------------------
    def _sentence(self, sentence: str, index: int) -> str:
        rng, lvl = self.rng, self.intensity
        m = re.match(r"^(.*?)([.!?]*)$", sentence, re.S)
        body, punct = m.group(1).strip(), m.group(2)
        question = "?" in punct
        s = body.lower().replace("’", "'")

        if lvl >= 1:
            s = _COPULA_RE.sub(_copula, s)
        for pattern, repl in _COMPILED_PHRASES:
            s = pattern.sub(repl, s)

        words_out: list[str] = []
        prev = ""
        for tok in _TOKEN_RE.findall(s):
            if not re.match(r"[A-Za-z']", tok):
                words_out.append(tok)
                continue
            word = tok
            if lvl >= 1 and word in ("is", "are", "am"):
                continue  # Patois drops the copula: "di bwoy dem real mad"
            if word in _WORDS and (lvl >= 1 or self._is_light(word)):
                new = _pick(_WORDS[word], rng)
            elif lvl >= 1 or word.startswith("th"):
                new = _phonetic(word, lvl, rng)
            else:
                new = word
            # plural marker: "di bwoys" -> "di bwoy dem"
            if lvl >= 2 and prev == "di" and re.fullmatch(r"[a-z]{2,}[^su]s", word) and word not in _WORDS:
                new = _phonetic(word[:-1], lvl, rng) + " dem"
            words_out.append(new)
            prev = new
        s = "".join(words_out)
        s = re.sub(r"\s+([,;:])", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\b(\w+) \1 \1\b", r"\1 \1", s)  # avoid accidental triple repeats

        # Flavour: openers and tags, more of them the higher the intensity
        closed = False
        if s:
            opener_chance = (0.08, 0.25, 0.4, 0.7)[lvl]
            if self.spicy and lvl >= 1 and rng.random() < (0.1, 0.2, 0.3, 0.45)[lvl]:
                s = f"{rng.choice(_SPICY_OPENERS)} {s}"
            elif index == 0 and rng.random() < opener_chance:
                s = f"{rng.choice(_OPENERS[lvl])} {s}"
            tag_chance = (0.1, 0.3, 0.45, 0.75)[lvl]
            if not question and rng.random() < tag_chance:
                tag = rng.choice(_TAGS[lvl])
                if tag:
                    s = s + tag
                    if tag[-1] in "?!":
                        punct = ""
                        closed = True
        if not closed and lvl >= 3 and not question and not punct.endswith("!"):
            punct = "!"
        s = s[:1].upper() + s[1:] if s else s
        s = re.sub(r"([!?] )([a-z])", lambda mm: mm.group(1) + mm.group(2).upper(), s)
        s = re.sub(r"\b(jah|lawd)\b", lambda mm: mm.group(1).capitalize(), s)
        if closed:
            return s
        return s + (punct or ".")

    @staticmethod
    def _is_light(word: str) -> bool:
        """Swaps applied even at Tourist level."""
        return word in {"the", "this", "that", "friend", "man", "yes", "yeah", "hello", "hi", "good", "little", "thing", "think"}

    def segment(self, patois: str) -> list[Segment]:
        """Split Patois text into prosody segments with Jamaican-style delivery.

        Jamaican English is syllable-timed with a wide pitch range: phrases tend
        to rise through the middle and *fall hard* at the end of a statement,
        questions swing up, and stressed words get pitched up and stretched.
        """
        rng = self.rng
        chunks = [c.strip() for c in re.split(r"(?<=[,.!?;:])\s+", patois) if c.strip()]
        segments: list[Segment] = []
        for i, chunk in enumerate(chunks):
            words = re.findall(r"[A-Za-z']+", chunk.lower())
            loud = chunk.isupper() and len(chunk) > 3
            hits = sum(1 for w in words if w in _INTENSIFIERS)
            emphasis = min(3, hits + (2 if loud else 0) + (1 if chunk.endswith("!") else 0))
            if self.intensity >= 3:
                emphasis = min(3, emphasis + 1)
            last = i == len(chunks) - 1
            if chunk.endswith("?"):
                contour = "rise"
            elif chunk.endswith("!"):
                contour = "high" if emphasis >= 2 else "rise_fall"
            elif last or chunk.endswith("."):
                contour = "rise_fall"
            elif chunk.endswith(","):
                contour = rng.choice(("rise", "rise", "high"))
            else:
                contour = "flat"
            pause = 260 if chunk[-1:] in ".!?" else 110
            segments.append(Segment(text=chunk, emphasis=emphasis, contour=contour, pause_after_ms=pause))
        return segments


AD_LIBS = [
    "Ya mon!", "Wah gwaan!", "Big up yuhself!", "Bwoy!", "Lawd have mercy!", "Respect!", "Wicked!",
    "Bless up!", "Pull up, selecta!", "Wheel an' come again!", "Mi seh!", "Irie!", "Seen!",
    "One love!", "Nuff respect!", "Yow, yow, yow!", "Gwaan go do it!", "Jah know!", "Rahtid!",
    "Bad tune dat!", "Inna di morrows!", "Mi a go dead wid laugh!",
]
SPICY_AD_LIBS = ["Bomboclaat!", "Raasclaat!", "Bloodclaat!", "Wah di bomboclaat!"]


def translate(text: str, intensity: int = 2, spicy: bool = False, seed: int | None = None) -> Translation:
    return PatoisTranslator(intensity=intensity, spicy=spicy, seed=seed).translate(text)
