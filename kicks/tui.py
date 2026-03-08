"""Textual TUI for the kick drum synthesizer — Synthwave edition."""

import os
import subprocess
import tempfile
import threading

import numpy as np
import soundfile as sf
import torch
from rich.text import Text
from sklearn.decomposition import PCA
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Footer, Label, Static

from kicks import KickDataset, KickDataloader, VAE
from kicks.cluster import extract_latents, compute_descriptors
from kicks.config import get_device, BEST_CHECKPOINT, N_PCS
from kicks.model import SAMPLE_RATE
from kicks.vocoder import load_vocoder, spec_to_audio

# ── Synthwave colour palette ─────────────────────────────────────────

_BG_DEEP = "#0d0221"
_BG_PANEL = "#150533"
_NEON_CYAN = "#00fff2"
_NEON_PINK = "#ff2975"
_NEON_MAGENTA = "#f72585"
_NEON_PURPLE = "#b026ff"
_NEON_ORANGE = "#ff6b35"
_NEON_YELLOW = "#ffbe0b"
_DIM_PURPLE = "#3a1568"
_DIM_CYAN = "#0a4f4c"
_GRID_COLOR = "#2a1050"
_TEXT_BRIGHT = "#e0d0ff"
_TEXT_DIM = "#6a5a8a"

# Waveform gradient: cyan -> magenta (by intensity)
_WAVE_GRADIENT = ["#0a4f4c", "#00c9b7", "#00fff2", "#a855f7", "#d946ef", "#ff2975"]
# Spectrogram heat: deep purple -> magenta -> orange -> yellow (sunset)
_SPEC_GRADIENT = ["#0d0221", "#2a1050", "#6b21a8", "#c026d3", "#f72585", "#ff6b35", "#ffbe0b", "#fffb47"]

# ── ASCII art header ─────────────────────────────────────────────────

_LOGO = r"""
 ██ ▄█▀ ██▓ ▄████▄   ██ ▄█▀  ██████
 ██▄█▒ ▓██▒▒██▀ ▀█  ██▄█▒ ▒██    ▒
▓███▄░ ▒██▒▒▓█    ▄▓███▄░ ░ ▓██▄
▓██ █▄ ░██░▒▓▓▄ ▄██▓██ █▄   ▒   ██▒
▒██▒ █▄░██░▒ ▓███▀ ▒██▒ █▄▒██████▒▒
▒ ▒▒ ▓▒░▓  ░ ░▒ ▒  ▒ ▒▒ ▓▒▒ ▒▓▒ ▒ ░
"""

_SUN = r"""        ·  ·  ·  · ╱────────────╲ ·  ·  ·  ·
          ·  ·  ╱────────────────╲  ·  ·
"""

# ── Waveform visualisation ───────────────────────────────────────────

WAVEFORM_WIDTH = 72
WAVEFORM_HEIGHT = 9

_UPPER_BLOCKS = " ▁▂▃▄▅▆▇█"


def _color_at_intensity(val: float, gradient: list[str]) -> str:
    """Pick a colour from a gradient list based on value 0..1."""
    idx = min(int(val * (len(gradient) - 1)), len(gradient) - 1)
    return gradient[max(0, idx)]


def _render_waveform_rich(samples: np.ndarray, width: int = WAVEFORM_WIDTH, height: int = WAVEFORM_HEIGHT) -> Text:
    """Render a coloured waveform using Rich Text with neon gradient."""
    text = Text()

    if len(samples) == 0:
        for _ in range(height):
            text.append(" " * width + "\n")
        return text

    indices = np.linspace(0, len(samples) - 1, width).astype(int)
    cols = samples[indices]
    half = height // 2

    # Upper half
    for row in range(half, 0, -1):
        for c in range(width):
            val = max(0.0, float(cols[c]))
            fill = val * half - (row - 1)
            fill = max(0.0, min(1.0, fill))
            idx = int(fill * 8)
            char = _UPPER_BLOCKS[idx]
            if idx > 0:
                color = _color_at_intensity(val, _WAVE_GRADIENT)
                text.append(char, style=color)
            else:
                text.append(" ")
        text.append("\n")

    # Centre line
    for c in range(width):
        intensity = abs(float(cols[c]))
        if intensity > 0.05:
            color = _color_at_intensity(intensity, _WAVE_GRADIENT)
            text.append("═", style=color)
        else:
            text.append("─", style=_DIM_PURPLE)
    text.append("\n")

    # Lower half
    for row in range(half):
        for c in range(width):
            val = abs(min(0.0, float(cols[c])))
            fill = val * half - row
            fill = max(0.0, min(1.0, fill))
            idx = int(fill * 8)
            if idx > 0:
                color = _color_at_intensity(val, _WAVE_GRADIENT)
                text.append(_UPPER_BLOCKS[idx], style=color)
            else:
                text.append(" ")
        text.append("\n")

    return text


# ── Spectrogram visualisation ────────────────────────────────────────

_SPEC_CHARS = " ░▒▓█"


def _render_spectrogram_rich(spec: np.ndarray, width: int = WAVEFORM_WIDTH, height: int = 16) -> Text:
    """Render a coloured mel spectrogram using Rich Text with sunset heat map."""
    text = Text()
    n_mels, n_frames = spec.shape
    rows_per_bin = n_mels / height
    cols_per_frame = n_frames / width

    smin, smax = spec.min(), spec.max()
    if smax - smin > 1e-6:
        spec_n = (spec - smin) / (smax - smin)
    else:
        spec_n = np.zeros_like(spec)

    for row in range(height):
        mel_lo = int((height - 1 - row) * rows_per_bin)
        mel_hi = int((height - row) * rows_per_bin)
        for col in range(width):
            fr_lo = int(col * cols_per_frame)
            fr_hi = max(fr_lo + 1, int((col + 1) * cols_per_frame))
            val = float(spec_n[mel_lo:mel_hi, fr_lo:fr_hi].mean())
            char_idx = int(val * (len(_SPEC_CHARS) - 1))
            char = _SPEC_CHARS[char_idx]
            if char_idx > 0:
                color = _color_at_intensity(val, _SPEC_GRADIENT)
                text.append(char, style=color)
            else:
                text.append(" ")
        text.append("\n")

    return text


# ── Custom widgets ───────────────────────────────────────────────────

_BAR_WIDTH = 22
_BAR_FILL = "━"
_BAR_EMPTY = "─"
_BAR_KNOB = "◆"

# Per-slider neon accents
_SLIDER_COLORS = [_NEON_CYAN, _NEON_PINK, _NEON_PURPLE, _NEON_ORANGE]
_SLIDER_COLORS_DIM = [_DIM_CYAN, "#4a1535", "#3a1568", "#5a2a10"]


class SliderBar(Static, can_focus=True):
    """A focusable neon horizontal slider bar."""

    DEFAULT_CSS = """
    SliderBar {
        height: 3;
        padding: 0 1;
    }
    """

    value: reactive[int] = reactive(50)

    class Changed(Message):
        def __init__(self, slider: "SliderBar", value: int) -> None:
            super().__init__()
            self.slider = slider
            self.value = value

    def __init__(self, pc_index: int, name: str, **kwargs):
        super().__init__(**kwargs)
        self.pc_index = pc_index
        self.pc_name = name

    def render(self) -> Text:
        val = self.value / 100.0
        filled = int(val * _BAR_WIDTH)
        empty = _BAR_WIDTH - filled
        ci = self.pc_index % len(_SLIDER_COLORS)
        active = self.has_focus

        neon = _SLIDER_COLORS[ci]
        dim = _SLIDER_COLORS_DIM[ci]

        text = Text()
        # Glow indicator for focused slider
        if active:
            text.append(" ▸ ", style=f"bold {neon}")
        else:
            text.append("   ", style=_TEXT_DIM)
        text.append(f"{self.pc_name:<8}", style=f"bold {neon}" if active else _TEXT_DIM)
        text.append(f" {val:.2f}  ", style=_TEXT_BRIGHT if active else _TEXT_DIM)
        text.append(_BAR_FILL * filled, style=f"bold {neon}" if active else dim)
        text.append(_BAR_KNOB, style=f"bold {neon}" if active else _TEXT_DIM)
        text.append(_BAR_EMPTY * empty, style=_DIM_PURPLE)
        return text

    def on_key(self, event) -> None:
        if event.key == "right":
            self.value = min(100, self.value + 2)
            self.post_message(self.Changed(self, self.value))
            event.stop()
        elif event.key == "left":
            self.value = max(0, self.value - 2)
            self.post_message(self.Changed(self, self.value))
            event.stop()
        elif event.key == "shift+right":
            self.value = min(100, self.value + 10)
            self.post_message(self.Changed(self, self.value))
            event.stop()
        elif event.key == "shift+left":
            self.value = max(0, self.value - 10)
            self.post_message(self.Changed(self, self.value))
            event.stop()

    def watch_value(self, new_value: int) -> None:
        self.refresh()


class LogoWidget(Static):
    """Synthwave ASCII logo."""

    DEFAULT_CSS = """
    LogoWidget {
        height: auto;
        padding: 0 0;
        content-align: center middle;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_mount(self) -> None:
        text = Text()
        for line in _LOGO.strip("\n").split("\n"):
            text.append(line, style=f"bold {_NEON_MAGENTA}")
            text.append("\n")
        self.update(text)


class SunWidget(Static):
    """Retrowave sun/horizon."""

    DEFAULT_CSS = """
    SunWidget {
        height: auto;
        padding: 0 0;
    }
    """

    def on_mount(self) -> None:
        text = Text()
        sun_colors = [_NEON_YELLOW, _NEON_ORANGE, _NEON_PINK, _NEON_MAGENTA, _NEON_PURPLE]
        lines = [
            "           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
            "         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
            "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
            "       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
            "        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░",
        ]
        for i, line in enumerate(lines):
            ci = min(i, len(sun_colors) - 1)
            text.append(line, style=sun_colors[ci])
            text.append("\n")
        # Horizon grid lines
        grid = "  ─ ─ ─ ─ ─ ─ ─ ─ ─ ╱─────────────────────╲─ ─ ─ ─ ─ ─ ─ ─ ─ ─"
        text.append(grid, style=_DIM_PURPLE)
        self.update(text)


class WaveformDisplay(Static):
    DEFAULT_CSS = """
    WaveformDisplay {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)

    def set_waveform(self, samples: np.ndarray) -> None:
        self.update(_render_waveform_rich(samples))


class SpectrogramDisplay(Static):
    DEFAULT_CSS = """
    SpectrogramDisplay {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)

    def set_spectrogram(self, spec: np.ndarray) -> None:
        self.update(_render_spectrogram_rich(spec))


class StatusLine(Static):
    DEFAULT_CSS = f"""
    StatusLine {{
        height: 1;
        dock: bottom;
        padding: 0 2;
        background: {_BG_DEEP};
        color: {_NEON_CYAN};
    }}
    """


class NeonLabel(Static):
    """A glowing section header."""

    DEFAULT_CSS = """
    NeonLabel {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(self, label: str, color: str = _NEON_CYAN, **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._color = color

    def on_mount(self) -> None:
        text = Text()
        text.append("  ", style=f"on {self._color}")
        text.append(f" {self._label}", style=f"bold {self._color}")
        self.update(text)


class GenCounter(Static):
    """Shows the generation count with a neon badge."""

    DEFAULT_CSS = f"""
    GenCounter {{
        height: 1;
        padding: 0 1;
        color: {_TEXT_DIM};
    }}
    """

    def set_count(self, n: int, desc: str = "") -> None:
        text = Text()
        text.append("  GEN ", style=f"bold {_NEON_PINK}")
        text.append(f"#{n:03d}", style=f"bold {_NEON_YELLOW}")
        if desc:
            text.append(f"  {desc}", style=_TEXT_DIM)
        self.update(text)


# ── Main App ─────────────────────────────────────────────────────────

class KicksApp(App):
    """TUI kick drum synthesizer — Synthwave edition."""

    TITLE = "K I C K S"
    SUB_TITLE = "synthwave kick machine"

    CSS = f"""
    Screen {{
        background: {_BG_DEEP};
    }}

    Header {{
        background: {_BG_DEEP};
        color: {_NEON_MAGENTA};
        text-style: bold;
    }}

    #main-container {{
        layout: horizontal;
        height: 1fr;
    }}

    #left-panel {{
        width: 40;
        min-width: 36;
        border-right: thick {_DIM_PURPLE};
        padding: 1 0;
        background: {_BG_PANEL};
    }}

    #right-panel {{
        width: 1fr;
        padding: 1;
        background: {_BG_DEEP};
    }}

    #viz-container {{
        height: auto;
        margin: 0 0 1 0;
    }}

    #spec-container {{
        height: auto;
        margin: 0 0 1 0;
    }}

    #gen-counter {{
        height: 1;
        margin: 1 0 0 0;
    }}

    #logo {{
        height: auto;
        margin: 0 0 0 0;
        content-align: center middle;
    }}

    #sun {{
        height: auto;
        margin: 0 0 1 0;
    }}

    Footer {{
        background: {_BG_DEEP};
        color: {_NEON_CYAN};
    }}

    Footer > .footer--highlight {{
        background: {_DIM_PURPLE};
    }}

    Footer > .footer--key {{
        background: {_DIM_PURPLE};
        color: {_NEON_PINK};
    }}
    """

    BINDINGS = [
        Binding("space", "generate", "Generate", priority=True),
        Binding("p", "play", "Play"),
        Binding("s", "save", "Save WAV"),
        Binding("r", "randomize", "Randomize"),
        Binding("0", "reset_sliders", "Reset"),
        Binding("q", "quit", "Quit"),
    ]

    _loaded: reactive[bool] = reactive(False)
    _last_audio: np.ndarray | None = None
    _last_spec: np.ndarray | None = None
    _gen_count: int = 0
    _playing: bool = False

    def __init__(self, data_dir: str = "data/kicks"):
        super().__init__()
        self._data_dir = data_dir
        self._device: torch.device | None = None
        self._dataset: KickDataset | None = None
        self._model: VAE | None = None
        self._vocoder = None
        self._pca: PCA | None = None
        self._pc_projected: np.ndarray | None = None
        self._pc_names: list[str] = []
        self._pc_mins: list[float] = []
        self._pc_maxs: list[float] = []

    def compose(self) -> ComposeResult:
        with Container(id="main-container"):
            with Vertical(id="left-panel"):
                yield LogoWidget(id="logo")
                yield NeonLabel("CONTROLS", color=_NEON_PINK)
                # Sliders mount after loading
                yield GenCounter(id="gen-counter")
            with Vertical(id="right-panel"):
                yield SunWidget(id="sun")
                yield NeonLabel("WAVEFORM", color=_NEON_CYAN)
                with Container(id="viz-container"):
                    yield WaveformDisplay(id="waveform")
                yield NeonLabel("SPECTROGRAM", color=_NEON_ORANGE)
                with Container(id="spec-container"):
                    yield SpectrogramDisplay(id="spectrogram")
        yield StatusLine("LOADING MODEL...", id="status-line")
        yield Footer()

    def on_mount(self) -> None:
        self._load_model()

    @work(thread=True)
    def _load_model(self) -> None:
        status = self.query_one("#status-line", StatusLine)

        self.call_from_thread(status.update, "LOADING DATASET...")
        self._device = get_device()
        self._dataset = KickDataset(self._data_dir)
        dataloader = KickDataloader(self._dataset, batch_size=32, shuffle=False)

        self.call_from_thread(status.update, "LOADING VAE...")
        self._model = VAE(latent_dim=32)
        checkpoint = torch.load(BEST_CHECKPOINT, map_location=self._device)
        self._model.load_state_dict(checkpoint["model"])
        self._model.to(self._device)
        self._model.eval()

        self.call_from_thread(status.update, "LOADING VOCODER...")
        self._vocoder = load_vocoder(self._device)

        self.call_from_thread(status.update, "COMPUTING PCA...")
        latents, spectrograms = extract_latents(self._model, dataloader, self._device)

        descriptors = [compute_descriptors(s) for s in spectrograms]
        desc_keys = ["sub", "punch", "click", "bright", "decay"]
        desc_name_map = {
            "sub": "Sub", "punch": "Punch", "click": "Click",
            "bright": "Bright", "decay": "Decay",
        }
        desc_arrays = {k: np.array([d[k] for d in descriptors]) for k in desc_keys}

        self._pca = PCA(n_components=N_PCS)
        self._pc_projected = self._pca.fit_transform(latents)

        self._pc_names = []
        used: set[str] = set()
        for i in range(N_PCS):
            pc_vals = self._pc_projected[:, i]
            pc_mean, pc_std = pc_vals.mean(), pc_vals.std()
            best_desc, best_corr = None, 0.0
            for dk in desc_keys:
                if dk in used:
                    continue
                dv = desc_arrays[dk]
                d_mean, d_std = dv.mean(), dv.std()
                if pc_std > 0 and d_std > 0:
                    corr = float(((pc_vals - pc_mean) * (dv - d_mean)).mean() / (pc_std * d_std))
                else:
                    corr = 0.0
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_desc = dk
            if best_desc and abs(best_corr) >= 0.15:
                used.add(best_desc)
                self._pc_names.append(desc_name_map.get(best_desc, best_desc.capitalize()))
                if best_corr < 0:
                    self._pca.components_[i] *= -1
                    self._pc_projected[:, i] *= -1
            else:
                self._pc_names.append(f"PC{i + 1}")

        self._pc_mins = [float(self._pc_projected[:, i].min()) for i in range(N_PCS)]
        self._pc_maxs = [float(self._pc_projected[:, i].max()) for i in range(N_PCS)]

        def _mount_sliders():
            panel = self.query_one("#left-panel")
            counter = self.query_one("#gen-counter", GenCounter)
            for i in range(N_PCS):
                panel.mount(SliderBar(i, self._pc_names[i], id=f"pc_slider_{i}"), before=counter)
            status.update("READY  //  SPACE generate  |  LEFT/RIGHT adjust  |  TAB switch")

        self.call_from_thread(_mount_sliders)
        self._loaded = True

    def _get_pc_values(self) -> list[float]:
        vals = []
        for i in range(N_PCS):
            try:
                slider = self.query_one(f"#pc_slider_{i}", SliderBar)
                vals.append(slider.value / 100.0)
            except Exception:
                vals.append(0.5)
        return vals

    def action_generate(self) -> None:
        if not self._loaded:
            return
        self._do_generate()

    @work(thread=True)
    def _do_generate(self) -> None:
        status = self.query_one("#status-line", StatusLine)
        self.call_from_thread(status.update, "GENERATING...")

        raw_values = self._get_pc_values()
        pc_values = []
        for i, raw in enumerate(raw_values):
            val = self._pc_mins[i] + raw * (self._pc_maxs[i] - self._pc_mins[i])
            val = max(self._pc_mins[i], min(self._pc_maxs[i], val))
            pc_values.append(val)

        z_np = self._pca.inverse_transform([pc_values])
        z = torch.tensor(z_np, dtype=torch.float32).to(self._device)

        with torch.no_grad():
            spec = self._model.decode(z)
            waveform = spec_to_audio(spec, self._dataset, self._vocoder, self._device)

        audio_np = waveform.squeeze(0).numpy()
        self._last_audio = audio_np
        self._gen_count += 1

        from kicks.dataset import KickDataset as KDS
        spec_viz = KDS.denormalize(spec.cpu()).squeeze(0).squeeze(0).numpy()
        self._last_spec = spec_viz

        gen_count = self._gen_count
        desc = "  ".join(f"{self._pc_names[i]}={raw_values[i]:.2f}" for i in range(N_PCS))

        def _update_viz():
            self.query_one("#waveform", WaveformDisplay).set_waveform(audio_np)
            self.query_one("#spectrogram", SpectrogramDisplay).set_spectrogram(spec_viz)
            self.query_one("#gen-counter", GenCounter).set_count(gen_count, desc)
            status.update(f"READY  //  SPACE generate  |  LEFT/RIGHT adjust  |  TAB switch")

        self.call_from_thread(_update_viz)
        self._play_audio(audio_np)

    def _play_audio(self, audio: np.ndarray) -> None:
        if self._playing:
            return

        def _play():
            self._playing = True
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, SAMPLE_RATE)
                    tmp = f.name
                subprocess.run(["afplay", tmp], check=True, capture_output=True)
            except Exception:
                pass
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except Exception:
                        pass
                self._playing = False

        threading.Thread(target=_play, daemon=True).start()

    def action_play(self) -> None:
        if self._last_audio is not None:
            self._play_audio(self._last_audio)

    def action_save(self) -> None:
        if self._last_audio is None:
            return
        os.makedirs("output", exist_ok=True)
        path = f"output/tui_{self._gen_count}.wav"
        sf.write(path, self._last_audio, SAMPLE_RATE)
        status = self.query_one("#status-line", StatusLine)
        text = Text()
        text.append("SAVED ", style=f"bold {_NEON_CYAN}")
        text.append(path, style=f"bold {_NEON_YELLOW}")
        status.update(text)

    def action_randomize(self) -> None:
        if not self._loaded:
            return
        import random
        for i in range(N_PCS):
            try:
                slider = self.query_one(f"#pc_slider_{i}", SliderBar)
                slider.value = random.randint(0, 100)
            except Exception:
                pass
        self.action_generate()

    def action_reset_sliders(self) -> None:
        if not self._loaded:
            return
        for i in range(N_PCS):
            try:
                slider = self.query_one(f"#pc_slider_{i}", SliderBar)
                slider.value = 50
            except Exception:
                pass


def run_tui(data_dir: str = "data/kicks") -> None:
    """Entry point for the TUI."""
    app = KicksApp(data_dir=data_dir)
    app.run()
