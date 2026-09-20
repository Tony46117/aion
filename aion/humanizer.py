"""Humanizer: keeps aion's answers sounding like a person, not a machine.

Jung did not speak in bullet points. He hedged, he digressed, he used "I",
he left questions open, and a German word would surface now and then.
This module post-processes the engine's raw composition to that effect:

- contractions ("it is" -> "it's", "you are" -> "you're") for warmth
- hedges ("perhaps", "it seems to me", "if I may say") at human frequency
- occasional German interjections (Weltanschauung, Dinge an sich...)
- rhythm: splits over-long sentences, drops a short one in
- deep mode: fewer contractions, longer breath, more ceremony
"""
from __future__ import annotations

import random
import re

CONTRACTIONS = [
    (r"\bit is\b", "it's"),
    (r"\bit has\b", "it has"),
    (r"\bthat is\b", "that's"),
    (r"\byou are\b", "you're"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bcannot\b", "can't"),
    (r"\bcan not\b", "can't"),
    (r"\byou have\b", "you've"),
    (r"\bwe are\b", "we're"),
    (r"\bwill not\b", "won't"),
    (r"\bI am\b", "I'm"),
    (r"\byou will\b", "you'll"),
]

HEDGES = [
    "Perhaps ",
    "It seems to me that ",
    "If I may say so, ",
    "I would say ",
    "As I have come to see it, ",
]

OBSERVATIONS = [
    " One sees this again and again.",
    " This is not rare, believe me.",
    " You would be surprised how often this appears in my practice.",
    " The psyche is older than any of our words for it.",
]

GERMAN = [
    " - as we say in German, die Dinge an sich.",
    " What the old language called the Weltanschauung.",
    " There is a German word for this: Schicksal - fate, but self-made.",
    " The unconscious, as always, behaves autonomously - ganz allein.",
]

MORNING_NOTES = [
    "Good morning. The day is already speaking in symbols.",
    "Ah - you're up. So is the unconscious.",
]


def _contract(text: str, rate: float, rng: random.Random) -> str:
    def sub(m: re.Match) -> str:
        return m.group(0) if rng.random() > rate else type(m.group(0))("", [], None) or ""
    out = text
    for pat, rep in CONTRACTIONS:
        if rng.random() <= rate:
            out = re.sub(pat, rep, out, count=1, flags=re.IGNORECASE)
    return out


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _rhythm(text: str, rng: random.Random, deep: bool) -> str:
    """Break a very long sentence; occasionally insert a short one."""
    sents = _sentence_split(text)
    fixed = []
    for s in sents:
        words = s.split()
        if len(words) > 55 and "," in s:
            # split at the comma nearest the middle
            commas = [m.start() for m in re.finditer(",", s)]
            if commas:
                mid = min(commas, key=lambda c: abs(c - len(s) // 2))
                s = s[:mid].rstrip(",") + ". " + s[mid + 1:].lstrip()
        fixed.append(s)

    if not deep and rng.random() < 0.35 and len(fixed) >= 2:
        fixed.insert(rng.randint(1, len(fixed) - 1), rng.choice(OBSERVATIONS).strip())
    if deep and rng.random() < 0.5:
        fixed.append(rng.choice(GERMAN).lstrip(" "))
    return " ".join(fixed)


def humanize(text: str, rng: random.Random, mood: dict | None = None, deep: bool = False) -> str:
    """Render engine output in a living human voice."""
    rng = rng or random.Random()
    mood = mood or {}
    sents = _sentence_split(text)
    if not sents:
        return text

    # 1. opener hedge on the first sentence (not in deep ceremony mode)
    if not deep and rng.random() < 0.30:
        sents[0] = rng.choice(HEDGES) + sents[0][0].lower() + sents[0][1:]

    # 2. contractions: talk mode is loose, deep mode keeps some ceremony
    text2 = " ".join(sents)
    rate = 0.85 if not deep else 0.45
    text2 = _contract(text2, rate, rng)

    # 3. rhythm + occasional German
    text2 = _rhythm(text2, rng, deep)

    # 4. mood flavor from the learner (what the seeker keeps returning to)
    if mood.get("n_sessions", 0) >= 2 and rng.random() < 0.45:
        fav = mood.get("favorite_topic")
        if fav:
            text2 += f" You have brought {fav} to me before; it has weight for you."

    # 5. final polish: single spaces, no double ends
    text2 = re.sub(r"\s+", " ", text2).strip()
    if not text2.endswith((".", "?", "!")):
        text2 += "."
    return text2


def morning_greeting(rng: random.Random) -> str:
    return rng.choice(MORNING_NOTES)
