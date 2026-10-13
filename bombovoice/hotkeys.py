"""Global hotkeys (work while your game or Discord has focus). Optional: pip install pynput."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class Hotkeys:
    def __init__(self, engine, mapping: dict[str, str]):
        self.engine = engine
        self.mapping = mapping
        self.listener = None

    def start(self) -> bool:
        try:
            from pynput import keyboard
        except Exception as exc:  # not installed, or no display server
            log.info("Global hotkeys disabled (%s). pip install pynput to enable.", exc)
            return False
        bindings = {combo: (lambda a=action: self.engine.action(a)) for action, combo in self.mapping.items() if combo}
        try:
            self.listener = keyboard.GlobalHotKeys(bindings)
            self.listener.start()
        except Exception as exc:
            log.warning("Could not register hotkeys: %s", exc)
            return False
        return True

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def describe(self) -> str:
        return "  ".join(f"{combo.strip('<>').upper()}={action}" for action, combo in self.mapping.items() if combo)
