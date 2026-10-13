import json

from bombovoice.config import SETTINGS_VERSION, Settings


def test_old_settings_get_new_voice_defaults_but_keep_user_choices(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"rasp": 0.25, "warmth_db": 4.0, "quality": "hq", "base_pitch": -4.0, "intensity": 3}))
    s = Settings.load(p)
    assert s.rasp == Settings().rasp and s.quality == "natural"
    assert s.base_pitch == -4.0 and s.intensity == 3
    assert s.settings_version == SETTINGS_VERSION
    s.rasp = 0.5
    s.save(p)
    assert Settings.load(p).rasp == 0.5  # once upgraded, user changes stick
