"""Simulates Windows: the same mic appears under MME, DirectSound and WASAPI, and MME refuses to open."""

from types import SimpleNamespace

import pytest

from bombovoice import devices

APIS = [
    {"name": "MME", "devices": [0, 1, 4], "default_input_device": 0, "default_output_device": 1},
    {"name": "Windows DirectSound", "devices": [2], "default_input_device": 2, "default_output_device": -1},
    {"name": "Windows WASAPI", "devices": [3, 5], "default_input_device": 3, "default_output_device": 5},
]
DEVS = [
    {"name": "Microphone (Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Speakers (Realtek(R) Audio)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Microphone (Realtek(R) Audio)", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "Microphone (Realtek(R) Audio)", "hostapi": 2, "max_input_channels": 2, "max_output_channels": 0},
    {"name": "CABLE Input (VB-Audio Virtual C", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "CABLE Input (VB-Audio Virtual Cable)", "hostapi": 2, "max_input_channels": 0, "max_output_channels": 2},
]


class FakeStream:
    opened = []

    def __init__(self, device, channels, samplerate, blocksize, dtype, callback=None, latency=None, extra_settings=None):
        self.device = device
        self.fail(device, channels, extra_settings)
        FakeStream.opened.append((device, channels, latency, extra_settings))

    def fail(self, device, channels, extra):
        raise NotImplementedError

    def close(self):
        pass


def fake_sd(failing_apis=("MME",), wasapi_needs_stereo=False, everything_fails=False):
    class Stream(FakeStream):
        def fail(self, device, channels, extra):
            api = APIS[DEVS[device]["hostapi"]]["name"]
            if everything_fails or api in failing_apis:
                raise RuntimeError("Error opening InputStream: Unanticipated host error [PaErrorCode -9999]: "
                                   "'Undefined external error.' [MME error 1]")
            if api == "Windows WASAPI" and wasapi_needs_stereo and channels == 1:
                raise RuntimeError("Invalid number of channels")

    return SimpleNamespace(
        query_devices=lambda: DEVS, query_hostapis=lambda: APIS,
        default=SimpleNamespace(device=[0, 1]),
        InputStream=Stream, OutputStream=Stream,
        WasapiSettings=lambda auto_convert=False: ("wasapi", auto_convert),
    )


@pytest.fixture(autouse=True)
def _reset():
    FakeStream.opened.clear()


def test_default_mic_prefers_wasapi_with_auto_convert(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: fake_sd())
    st, idx, desc = devices.open_stream("input", None, 48000, 256, None)
    assert idx == 3 and "WASAPI" in desc
    assert FakeStream.opened[-1][3] == ("wasapi", True)


def test_chosen_mme_mic_falls_back_to_same_mic_elsewhere(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: fake_sd(failing_apis=("MME", "Windows WASAPI")))
    st, idx, desc = devices.open_stream("input", 0, 48000, 256, None)
    assert idx == 2 and "DirectSound" in desc


def test_stereo_fallback(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: fake_sd(wasapi_needs_stereo=True))
    st, idx, desc = devices.open_stream("input", 3, 48000, 256, None)
    assert idx == 3 and FakeStream.opened[-1][1] == 2


def test_truncated_mme_name_matches_full_name(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: fake_sd())
    assert devices.stream_candidates("output", 4) == [4, 5]


def test_total_failure_explains_the_privacy_setting(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: fake_sd(everything_fails=True))
    with pytest.raises(devices.AudioOpenError) as info:
        devices.open_stream("input", None, 48000, 256, None)
    msg = str(info.value)
    assert "Let desktop apps access your microphone" in msg and "MME error 1" in msg


def test_device_test_reports_each_entry(monkeypatch):
    monkeypatch.setattr(devices, "_sd", lambda: fake_sd())
    results = devices.test_devices()
    by_idx = {(k, i): r for k, i, _, r in results}
    assert by_idx[("input", 0)].startswith("FAILED") and by_idx[("input", 3)] == "OK"
