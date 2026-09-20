"""Jung's voice: offline TTS via espeak-ng, shaped to sound like him.

The espeak-ng voice is slowed and pitched down so the delivery has the
measured, accented, deliberate cadence Jung had in interviews. Voice
preference is stored in ``~/.aion/voice.json`` so /voice persists.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

MEM_DIR = Path.home() / ".aion"
VOICE_PREF = MEM_DIR / "voice.json"


def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _binary() -> str:
    return shutil.which("espeak-ng") or shutil.which("espeak") or "espeak-ng"


# Voice profile: slowed cadence, lowered pitch, accented delivery.
PROFILES = {
    "jung":    ["-v", "de+slow", "-p", "25", "-s", "128", "-g", "12"],
    "deep":    ["-v", "de+slow", "-p", "15", "-s", "110", "-g", "18"],
    "neutral": ["-v", "en-us", "-p", "40", "-s", "160", "-g", "8"],
}


def load_pref() -> str:
    try:
        return json.loads(VOICE_PREF.read_text()).get("profile", "jung")
    except Exception:
        return "jung"


def save_pref(profile: str) -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_PREF.write_text(json.dumps({"profile": profile}))


def speak(text: str, profile: str | None = None, deep: bool = False) -> None:
    """Speak text aloud; falls back silently if espeak-ng is missing."""
    if not espeak_available():
        return
    profile = profile or load_pref()
    if deep and profile == "jung":
        profile = "deep"
    # Clean spoken text: strip quotes, symbols, markdown artifacts.
    spoken = re.sub(r"[\"*_#>`]", "", text)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    args = [_binary(), *PROFILES.get(profile, PROFILES["jung"]), spoken]
    try:
        subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    except Exception:
        pass


def stop() -> None:
    try:
        subprocess.run(["pkill", "-f", _binary()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
