"""Tkinter desktop app: routing, live controls, soundboard, and a Patois log."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import messagebox, ttk

from .config import MODES, Settings
from .patois import INTENSITY_NAMES, PatoisTranslator

log = logging.getLogger(__name__)

GREEN, GOLD, BLACK = "#009B3A", "#FED100", "#111111"
BG, PANEL, TEXT, MUTED = "#151515", "#1f1f1f", "#f2f2f2", "#9a9a9a"

SOUNDBOARD = [
    ("AIR HORN", "airhorn"), ("DUB SIREN", "siren"), ("REWIND!", "rewind"), ("BASS DROP", "bassdrop"),
    ("RIDDIM", "riddim"), ("AD-LIB", "adlib"), ("DUB ECHO", "echo"), ("PEW PEW", "laser"),
]

ENGINES = {
    "natural": "Natural (PSOLA) - recommended",
    "pv": "Smooth (phase vocoder)",
    "fast": "Low latency (delay line)",
}

SLIDERS = [
    # key, label, from, to, resolution
    ("base_pitch", "Deepness (semitones)", -8.0, 4.0, 0.5),
    ("formant", "Big-man chest (formant)", -5.0, 3.0, 0.5),
    ("lilt", "Sing-song lilt", 0.0, 2.0, 0.05),
    ("rasp", "Gravel / rasp", 0.0, 1.0, 0.05),
    ("warmth_db", "Warmth (dB)", 0.0, 10.0, 0.5),
    ("mic_gain_db", "Mic gain (dB)", -12.0, 18.0, 0.5),
    ("riddim_volume", "Riddim volume", 0.0, 0.6, 0.01),
    ("sfx_volume", "Soundboard volume", 0.0, 1.0, 0.05),
]


class App:
    def __init__(self, root: tk.Tk, settings: Settings):
        self.root = root
        self.s = settings
        self.events: queue.Queue = queue.Queue()
        self.engine = None
        self.hotkeys = None
        self.devices = []
        self._build()
        self._poll()

    # ------------------------------------------------------------------ layout
    def _build(self):
        r = self.root
        r.title("Bombovoice - Jamaican Voice Changer")
        r.configure(bg=BG)
        r.minsize(980, 640)
        style = ttk.Style(r)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TLabelframe", background=BG, foreground=GOLD)
        style.configure("TLabelframe.Label", background=BG, foreground=GOLD, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.configure("TRadiobutton", background=BG, foreground=TEXT)
        style.configure("TButton", background=PANEL, foreground=TEXT, padding=6)
        style.map("TButton", background=[("active", GREEN)])
        style.configure("Go.TButton", background=GREEN, foreground="white", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Horizontal.TScale", background=BG)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TEXT)],
                  selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", GOLD)])

        header = tk.Frame(r, bg=BLACK)
        header.pack(fill="x")
        tk.Label(header, text="BOMBOVOICE", bg=BLACK, fg=GOLD, font=("Segoe UI Black", 22, "bold")).pack(side="left", padx=14, pady=8)
        tk.Label(header, text="wah gwaan - make yuh voice sound like a real yaadie", bg=BLACK, fg=GREEN,
                 font=("Segoe UI", 11, "italic")).pack(side="left")
        flag = tk.Canvas(header, width=60, height=36, bg=BLACK, highlightthickness=0)
        flag.pack(side="right", padx=14)
        flag.create_rectangle(0, 0, 60, 36, fill=GREEN, width=0)
        flag.create_polygon(0, 0, 60, 36, 60, 0, fill=BLACK)
        flag.create_polygon(0, 0, 0, 36, 60, 36, fill=BLACK)
        flag.create_polygon(0, 36, 60, 0, 60, 36, fill=GREEN)
        flag.create_line(0, 0, 60, 36, fill=GOLD, width=6)
        flag.create_line(0, 36, 60, 0, fill=GOLD, width=6)

        body = tk.Frame(r, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y")
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # --- routing
        rt = ttk.LabelFrame(left, text="Routing")
        rt.pack(fill="x", pady=(0, 8))
        self.in_var, self.out_var, self.mon_var = tk.StringVar(), tk.StringVar(), tk.BooleanVar(value=self.s.monitor)
        ttk.Label(rt, text="Microphone").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.in_box = ttk.Combobox(rt, textvariable=self.in_var, width=42, state="readonly")
        self.in_box.grid(row=0, column=1, padx=6, pady=2)
        ttk.Label(rt, text="Send to").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.out_box = ttk.Combobox(rt, textvariable=self.out_var, width=42, state="readonly")
        self.out_box.grid(row=1, column=1, padx=6, pady=2)
        ttk.Checkbutton(rt, text="Monitor (hear yourself)", variable=self.mon_var,
                        command=lambda: self._set(monitor=self.mon_var.get())).grid(row=2, column=1, sticky="w", padx=6)
        btns = tk.Frame(rt, bg=BG)
        btns.grid(row=3, column=0, columnspan=2, sticky="ew", pady=6, padx=6)
        self.go_btn = ttk.Button(btns, text="▶  START", style="Go.TButton", command=self.toggle)
        self.go_btn.pack(side="left", fill="x", expand=True)
        ttk.Button(btns, text="Mic setup help", command=self._setup_help).pack(side="left", padx=(6, 0))
        self._refresh_devices()

        # --- mode
        md = ttk.LabelFrame(left, text="Mode")
        md.pack(fill="x", pady=(0, 8))
        self.mode_var = tk.StringVar(value=self.s.mode)
        for key, desc in MODES.items():
            ttk.Radiobutton(md, text=desc, value=key, variable=self.mode_var,
                            command=lambda: self._set(mode=self.mode_var.get())).pack(anchor="w", padx=6, pady=1)

        # --- accent
        ac = ttk.LabelFrame(left, text="Accent")
        ac.pack(fill="x", pady=(0, 8))
        self.int_var = tk.IntVar(value=self.s.intensity)
        self.int_label = ttk.Label(ac, text="")
        ttk.Label(ac, text="Intensity").grid(row=0, column=0, sticky="w", padx=6)
        self.int_scale = ttk.Scale(ac, from_=0, to=3, orient="horizontal", length=180, command=self._on_intensity)
        self.int_scale.grid(row=0, column=1, padx=6)
        self.int_label.grid(row=0, column=2, sticky="w")
        self.int_scale.set(self.s.intensity)
        self._on_intensity(self.s.intensity)
        ttk.Label(ac, text="AI character").grid(row=1, column=0, sticky="w", padx=6)
        self.style_var = tk.StringVar(value=self.s.style)
        sb = ttk.Combobox(ac, textvariable=self.style_var, state="readonly", width=14,
                          values=["yaadie", "dancehall", "rasta", "auntie", "commentator"])
        sb.grid(row=1, column=1, sticky="w", padx=6, pady=2)
        sb.bind("<<ComboboxSelected>>", lambda e: self._set(style=self.style_var.get()))
        self.spicy_var = tk.BooleanVar(value=self.s.spicy)
        ttk.Checkbutton(ac, text="Spicy (bomboclaat!)", variable=self.spicy_var,
                        command=lambda: self._set(spicy=self.spicy_var.get())).grid(row=2, column=0, columnspan=2, sticky="w", padx=6)
        self.ai_var = tk.BooleanVar(value=self.s.use_ai)
        ttk.Checkbutton(ac, text="Use Claude AI director  via", variable=self.ai_var,
                        command=lambda: self._set(use_ai=self.ai_var.get())).grid(row=3, column=0, columnspan=2, sticky="w", padx=6)
        self.provider_var = tk.StringVar(value=self.s.ai_provider)
        pv = ttk.Combobox(ac, textvariable=self.provider_var, state="readonly", width=14,
                          values=["auto", "openrouter", "anthropic"])
        pv.grid(row=3, column=2, sticky="w", padx=6)
        pv.bind("<<ComboboxSelected>>", lambda e: self._set(ai_provider=self.provider_var.get()))
        self.auto_var = tk.BooleanVar(value=self.s.auto_adlib)
        ttk.Checkbutton(ac, text="Random ad-libs every ~25s", variable=self.auto_var,
                        command=lambda: self._set(auto_adlib=self.auto_var.get())).grid(row=4, column=0, columnspan=2, sticky="w", padx=6)

        # --- voice sliders
        vc = ttk.LabelFrame(left, text="Voice")
        vc.pack(fill="x")
        self.slider_labels = {}
        ttk.Label(vc, text="Voice engine").grid(row=0, column=0, sticky="w", padx=6)
        q = self.s.quality if self.s.quality in ENGINES else "natural"
        self.engine_var = tk.StringVar(value=ENGINES[q])
        eb = ttk.Combobox(vc, textvariable=self.engine_var, state="readonly", width=26, values=list(ENGINES.values()))
        eb.grid(row=0, column=1, columnspan=2, sticky="w", padx=6, pady=2)
        eb.bind("<<ComboboxSelected>>", lambda e: self._set(
            quality={v: k for k, v in ENGINES.items()}[self.engine_var.get()]))
        for i, (key, label, lo, hi, res) in enumerate(SLIDERS, start=1):
            ttk.Label(vc, text=label).grid(row=i, column=0, sticky="w", padx=6)
            lbl = ttk.Label(vc, text=f"{getattr(self.s, key):.2f}", width=6)
            lbl.grid(row=i, column=2)
            self.slider_labels[key] = lbl
            scale = ttk.Scale(vc, from_=lo, to=hi, orient="horizontal", length=180,
                              command=lambda v, k=key, rs=res: self._on_slider(k, v, rs))
            scale.set(getattr(self.s, key))
            scale.grid(row=i, column=1, padx=6)

        # --- soundboard
        sbf = ttk.LabelFrame(right, text="Sound system  (F1-F11 hotkeys work in-game)")
        sbf.pack(fill="x")
        for i, (label, action) in enumerate(SOUNDBOARD):
            ttk.Button(sbf, text=label, command=lambda a=action: self._action(a)).grid(
                row=i // 4, column=i % 4, sticky="ew", padx=4, pady=4)
        for c in range(4):
            sbf.columnconfigure(c, weight=1)

        # --- type & say
        ts = ttk.LabelFrame(right, text="Type it - Bombovoice seh it")
        ts.pack(fill="x", pady=8)
        self.say_var = tk.StringVar()
        entry = ttk.Entry(ts, textvariable=self.say_var)
        entry.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        entry.bind("<Return>", lambda e: self._say())
        ttk.Button(ts, text="Say it!", command=self._say).pack(side="left", padx=4)
        ttk.Button(ts, text="Preview", command=self._preview).pack(side="left", padx=(0, 6))

        # --- meter + log
        mf = tk.Frame(right, bg=BG)
        mf.pack(fill="x")
        ttk.Label(mf, text="Mic", style="Muted.TLabel").pack(side="left")
        self.meter = tk.Canvas(mf, height=12, bg=PANEL, highlightthickness=0)
        self.meter.pack(side="left", fill="x", expand=True, padx=6)
        self.meter_bar = self.meter.create_rectangle(0, 0, 0, 12, fill=GREEN, width=0)

        lf = ttk.LabelFrame(right, text="Wah dem a seh")
        lf.pack(fill="both", expand=True, pady=(8, 0))
        self.log = tk.Text(lf, bg=PANEL, fg=TEXT, wrap="word", font=("Consolas", 10), relief="flat", height=12)
        self.log.pack(fill="both", expand=True, padx=4, pady=4)
        self.log.tag_configure("you", foreground=MUTED)
        self.log.tag_configure("yaad", foreground=GOLD, font=("Consolas", 11, "bold"))
        self.log.tag_configure("status", foreground=GREEN)
        self.log.configure(state="disabled")

        self.status_var = tk.StringVar(value="Pick yuh mic and press START. Respect!")
        tk.Label(r, textvariable=self.status_var, bg=BLACK, fg=TEXT, anchor="w", padx=10).pack(fill="x", side="bottom")
        r.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------- behaviours
    def _refresh_devices(self):
        try:
            from .devices import find_virtual_output, list_devices

            self.devices = list_devices()
            virt = find_virtual_output()
        except Exception as exc:
            self.devices, virt = [], None
            self._log(f"Audio devices unavailable: {exc}", "status")
        ins = ["(system default)"] + [f"{d['index']}: {d['name']}  [{d['hostapi']}]" for d in self.devices if d["inputs"] > 0]
        outs = ["(auto: virtual cable)"] + [f"{d['index']}: {d['name']}  [{d['hostapi']}]" for d in self.devices if d["outputs"] > 0]
        self.in_box["values"], self.out_box["values"] = ins, outs
        self.in_var.set(self._match(ins, self.s.input_device))
        self.out_var.set(self._match(outs, self.s.output_device))
        if virt is None and self.devices:
            self._log("No virtual cable detected yet - click 'Mic setup help' so Discord/games can hear yuh.", "status")

    @staticmethod
    def _match(options, spec):
        if spec is None:
            return options[0]
        for o in options[1:]:
            if o.startswith(f"{spec}:") or str(spec).lower() in o.lower():
                return o
        return options[0]

    @staticmethod
    def _parse_choice(choice: str):
        return None if choice.startswith("(") else int(choice.split(":", 1)[0])

    def toggle(self):
        if self.engine and self.engine.running:
            self.engine.stop()
            if self.hotkeys:
                self.hotkeys.stop()
            self.go_btn.configure(text="▶  START")
            return
        self.s.input_device = self._parse_choice(self.in_var.get())
        self.s.output_device = self._parse_choice(self.out_var.get())
        try:
            if self.engine is None:
                from .engine import Engine

                self.engine = Engine(self.s, on_event=lambda k, p: self.events.put((k, p)))
                self.engine.prepare()
            self.engine.start()
        except Exception as exc:
            messagebox.showerror("Bombovoice", f"Could not start audio:\n\n{exc}")
            return
        from .hotkeys import Hotkeys

        self.hotkeys = Hotkeys(self.engine, self.s.hotkeys)
        if self.hotkeys.start():
            self._log("Hotkeys: " + self.hotkeys.describe(), "status")
        self.go_btn.configure(text="■  STOP")

    def _set(self, **changes):
        if self.engine:
            self.engine.set(**changes)
        else:
            for k, v in changes.items():
                setattr(self.s, k, v)

    def _on_slider(self, key, value, res):
        v = round(float(value) / res) * res
        self.slider_labels[key].configure(text=f"{v:.2f}")
        self._set(**{key: v})

    def _on_intensity(self, value):
        lvl = int(round(float(value)))
        self.int_label.configure(text=INTENSITY_NAMES[lvl])
        if lvl != self.s.intensity:
            self._set(intensity=lvl)

    def _action(self, name):
        if not self.engine or not self.engine.running:
            self.status_var.set("Press START first, bredda!")
            return
        self.engine.action(name)

    def _say(self):
        text = self.say_var.get().strip()
        if not text:
            return
        if not self.engine or not self.engine.running:
            self._preview()
            return
        self.engine.say(text)
        self.say_var.set("")

    def _preview(self):
        text = self.say_var.get().strip()
        if text:
            tr = PatoisTranslator(intensity=self.s.intensity, spicy=self.s.spicy).translate(text)
            self._log(f"you : {text}", "you")
            self._log(f"yaad: {tr.patois}   (offline preview)", "yaad")

    def _setup_help(self):
        from .devices import setup_instructions

        messagebox.showinfo("Use Bombovoice as your microphone", setup_instructions())
        self._refresh_devices()

    def _log(self, text, tag=None):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "heard":
                    self._log(f"you : {payload}", "you")
                elif kind == "patois":
                    self._log(f"yaad: {payload.patois}", "yaad")
                elif kind == "adlib":
                    self._log(f"      ~ {payload} ~", "yaad")
                elif kind == "status":
                    self.status_var.set(str(payload))
                    self._log(f"* {payload}", "status")
                elif kind == "sfx":
                    self.status_var.set(f"{payload}!")
        except queue.Empty:
            pass
        if self.engine and self.engine.running:
            w = self.meter.winfo_width()
            lvl = min(1.0, self.engine.voice_level * 8)
            self.meter.coords(self.meter_bar, 0, 0, int(w * lvl), 12)
            self.meter.itemconfigure(self.meter_bar, fill=GOLD if lvl > 0.8 else GREEN)
        self.root.after(50, self._poll)

    def on_close(self):
        try:
            self.s.save()
        except Exception as exc:
            log.warning("could not save settings: %s", exc)
        if self.hotkeys:
            self.hotkeys.stop()
        if self.engine:
            self.engine.close()
        self.root.destroy()


def run_gui(settings: Settings | None = None):
    root = tk.Tk()
    App(root, settings or Settings.load())
    root.mainloop()
