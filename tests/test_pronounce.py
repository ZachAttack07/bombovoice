import re

from bombovoice.patois import AD_LIBS, SPICY_AD_LIBS, translate
from bombovoice.pronounce import SAY_AS, speakable


def test_tricky_words_become_pronounceable():
    out = speakable("Im seh im cyaan come, mi tyad! Di bwoy dem a nyam.").lower()
    for bad in ("seh", "cyaan", "tyad", "bwoy", "nyam", " mi ", " di "):
        assert bad not in f" {out} "
    assert "say" in out and "kahn" in out


def test_no_all_caps_or_spell_out_traps():
    lines = AD_LIBS + SPICY_AD_LIBS + [
        translate("Hello my friend, I think this game is really crazy. Thank you so much!", 3, spicy=True, seed=s).patois
        for s in range(20)
    ]
    for line in lines:
        said = speakable(line)
        for word in re.findall(r"[A-Za-z][A-Za-z'\-]*", said):
            assert not (word.isupper() and len(word) > 1), (line, said)
            assert not re.fullmatch(r"[^aeiouyAEIOUY]+", word), (line, said)  # no vowel -> spelled out
            ok = {"yeah", "oh", "ah", "wah", "nah", "yah", "doh", "choh", "noh", "dah", "hah", "ooh", "rah"}
            assert not re.search(r"[eiu]h$", word.lower()) or word.lower() in ok, (line, said)


def test_every_mapping_has_a_vowel():
    for spelled in SAY_AS.values():
        assert re.search(r"[aeiouy]", spelled)


def test_display_text_is_unchanged_by_translation():
    tr = translate("He said he can't come", 2, seed=1)
    assert "seh" in tr.patois  # the screen still shows real Patois
