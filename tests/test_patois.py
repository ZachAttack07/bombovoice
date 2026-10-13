from bombovoice.patois import CONTOURS, PatoisTranslator, translate


def test_core_vocabulary():
    out = translate("I think the thing is with my friend", intensity=1, seed=0).patois.lower()
    for word in ("mi", "tink", "di", "ting", "wid"):
        assert word in out.split() or word in out


def test_grammar_progressive_and_copula():
    out = translate("I am going to the store.", intensity=1, seed=0).patois.lower()
    assert "mi a go" in out
    out = translate("The boys are really tired.", intensity=2, seed=0).patois.lower()
    assert "dem" in out and " are " not in f" {out} " and "tyad" in out


def test_negation():
    out = translate("I can't do that, I don't know.", intensity=1, seed=0).patois.lower()
    assert "cyaan" in out and "nuh know" in out


def test_tourist_is_lighter_than_maximum():
    text = "Hello my friend, I think this little thing is really good."
    light = translate(text, intensity=0, seed=4).patois
    heavy = translate(text, intensity=3, seed=4).patois
    assert light != heavy
    assert "my" in light.lower()  # tourist keeps most English


def test_spicy_only_when_enabled():
    text = ". ".join(["That is crazy"] * 30) + "."
    clean = translate(text, intensity=3, spicy=False, seed=1).patois.lower()
    spicy = translate(text, intensity=3, spicy=True, seed=1).patois.lower()
    assert "claat" not in clean
    assert "claat" in spicy


def test_segments_have_delivery():
    tr = translate("Oh my god, are you serious? That is so funny!", intensity=2, seed=2)
    assert tr.segments
    assert all(s.contour in CONTOURS for s in tr.segments)
    assert any(s.contour == "rise" for s in tr.segments)  # the question swoops up
    assert all(0 <= s.emphasis <= 3 for s in tr.segments)


def test_deterministic_with_seed():
    a = PatoisTranslator(intensity=2, seed=9).translate("What is going on my brother?")
    b = PatoisTranslator(intensity=2, seed=9).translate("What is going on my brother?")
    assert a.patois == b.patois


def test_empty():
    assert translate("   ").patois == ""
