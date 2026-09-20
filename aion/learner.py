"""Learner: aion's memory. Learns from every conversation, forever.

Every turn you exchange with aion is folded back into its memory:

- turns are persisted to ``~/.aion/memory.jsonl`` (durable across sessions)
- TF-IDF vectors are *incrementally refit* over remembered turns so future
  retrieval pulls in what you have actually talked about, not just the books
- a mood/topic profile accumulates (which Jungian themes you keep bringing,
  whether your questions skew anxious, angry, seeking...)
- what aion "remembers" bleeds back into answers through retrieval and the
  humanizer's mood flavor ("You have brought dreams to me before...")

The design is deliberately simple and honest: no pretending to be a large
language model, but a real statistical learner that measurably changes its
retrieval and tone as you talk with it.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MEM_DIR = Path.home() / ".aion"
MEMORY_FILE = MEM_DIR / "memory.jsonl"
PROFILE_FILE = MEM_DIR / "profile.json"
FACTS_FILE = MEM_DIR / "facts.json"

# ---------------------------------------------------------------------------
# Fact extraction: aion remembers who you are, not just what you said.
# ---------------------------------------------------------------------------

# Capitalized words after "I am" that are feelings, not names.
NAME_BLOCKLIST = {
    "tired", "angry", "sad", "happy", "lost", "afraid", "scared", "sorry",
    "sure", "fine", "good", "okay", "ok", "here", "back", "done", "ready",
    "confused", "depressed", "anxious", "worried", "sick", "ill", "lonely",
    "empty", "curious", "new", "old", "young", "angry", "stuck", "broken",
    "alive", "awake", "asleep", "drunk", "sober", "honest", "wrong", "right",
}

NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([a-z][a-z'\-]{1,20})\b", re.IGNORECASE),
    re.compile(r"\bcall me\s+([a-z][a-z'\-]{1,20})\b", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Z][a-z'\-]{1,20})\b"),   # case-sensitive on purpose
    re.compile(r"\bi'm\s+([A-Z][a-z'\-]{1,20})\b"),
]

# (key, regex, formatter) — values are phrased in the second person so they
# read naturally as "you told me once that you {value}". Later matches per key win.
FACT_EXTRACTORS = [
    ("age", re.compile(r"\bi(?:'m| am)\s+(\d{1,2})\s*(?:years old|yrs?\b|yo\b)", re.IGNORECASE),
     lambda m: f"are {m.group(1)} years old"),
    ("home", re.compile(r"\b(?i:i) (?:live in|am from|am living in)\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)"),
     lambda m: f"live in {m.group(1)}"),
    ("work", re.compile(r"\b(?i:i) work as\s+(an?\s+)?([a-z][a-z\- ]{2,25}?)(?:[.,;!\n]|$)", re.IGNORECASE),
     lambda m: f"work as {(m.group(1) or 'a ').strip()} {m.group(2).strip().lower()}"),
    ("work", re.compile(
        r"\bi(?:'m| am) an? (engineer|teacher|doctor|nurse|lawyer|student|artist|writer|"
        r"programmer|developer|designer|musician|scientist|therapist|analyst|accountant|"
        r"farmer|driver|chef|architect|journalist|manager|consultant|professor|lecturer|"
        r"priest|pastor|electrician|plumber|carpenter)\b", re.IGNORECASE),
     lambda m: f"are a {m.group(1).lower()}"),
    ("fear", re.compile(
        r"\bi(?:'m| am)\s+(?:afraid|scared|terrified|frightened)\s+of\s+([a-z][a-z ]{2,40}?)(?:[.,;!?\n]|$)",
        re.IGNORECASE),
     lambda m: f"fear {m.group(1).strip().lower()}"),
    ("dream", re.compile(
        r"\bi\s+(?:keep\s+)?dream(?:t|ed)?\s+(?:about|of)\s+([a-z][a-z ]{2,40}?)(?:[.,;!?\n]|$)",
        re.IGNORECASE),
     lambda m: f"have dreamed of {m.group(1).strip().lower()}"),
]

TOPIC_WORDS: dict[str, tuple[str, ...]] = {
    "your shadow": ("shadow", "dark", "hate", "enemy", "blame", "angry"),
    "dreams": ("dream", "nightmare", "sleep", "vision", "night"),
    "the anima": ("anima", "animus", "love", "woman", "man", "marriage", "lover"),
    "individuation": ("purpose", "whole", "become", "myself", "who am i", "career"),
    "the god-image": ("god", "religion", "sacred", "church", "prayer", "faith"),
    "the unconscious": ("unconscious", "psyche", "mind", "soul", "conscious"),
    "suffering": ("pain", "hurt", "depress", "anxiety", "anxious", "afraid", "fear", "sad"),
}

VIBES: dict[str, tuple[str, ...]] = {
    "anxious": ("afraid", "anxious", "worry", "panic", "scared", "nervous"),
    "sorrowful": ("sad", "grief", "loss", "cry", "lonely", "empty"),
    "searching": ("meaning", "why", "purpose", "who am i", "lost", "confused"),
    "struggling": ("angry", "hate", "fight", "battle", "cannot", "stuck"),
}


class Learner:
    """Persists conversation memory and refits retrieval over it."""

    def __init__(self, mem_dir: Path | None = None) -> None:
        self.mem_dir = mem_dir or MEM_DIR
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.mem_dir / "memory.jsonl"
        self.profile_file = self.mem_dir / "profile.json"

        self.turns: list[dict] = []
        self.facts: dict[str, str] = {}
        self._load()
        self._refit()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self.memory_file.exists():
            try:
                for line in self.memory_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        self.turns.append(json.loads(line))
            except Exception:
                self.turns = []
        self.profile: dict = {}
        if self.profile_file.exists():
            try:
                self.profile = json.loads(self.profile_file.read_text(encoding="utf-8"))
            except Exception:
                self.profile = {}
        if FACTS_FILE.exists() or (self.mem_dir / "facts.json").exists():
            try:
                self.facts = json.loads((self.mem_dir / "facts.json").read_text(encoding="utf-8"))
            except Exception:
                self.facts = {}

    def _persist(self) -> None:
        try:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.turns[-1], ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # learning
    # ------------------------------------------------------------------
    def record_turn(self, user_text: str, aion_text: str, deep: bool = False) -> None:
        entry = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deep": deep,
            "user": user_text,
            "aion": aion_text,
        }
        self.turns.append(entry)
        self._persist()
        self._update_profile(user_text)
        self._remember_facts(user_text)
        # refit periodically, not every keystroke
        if len(self.turns) % 5 == 0 or len(self.turns) < 10:
            self._refit()

    def learn(self, text: str) -> None:
        """Explicit teaching channel (/remember ...): fold a line into memory."""
        self._update_profile(text)
        self._remember_facts(text)

    def _update_profile(self, user_text: str) -> None:
        low = " " + user_text.lower() + " "
        prof = self.profile
        # topics
        topics = prof.setdefault("topics", {})
        for topic, words in TOPIC_WORDS.items():
            if any(w in low for w in words):
                topics[topic] = topics.get(topic, 0) + 1
        prof["topics"] = topics
        # vibes
        vibes = prof.setdefault("vibes", {})
        for vibe, words in VIBES.items():
            if any(w in low for w in words):
                vibes[vibe] = vibes.get(vibe, 0) + 1
        prof["vibes"] = vibes
        prof["n_user_turns"] = prof.get("n_user_turns", 0) + 1
        prof["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._save_profile()

    def _save_profile(self) -> None:
        try:
            self.profile_file.write_text(
                json.dumps(self.profile, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # fact memory: who the seeker is, remembered across sessions
    # ------------------------------------------------------------------
    def _remember_facts(self, text: str) -> None:
        changed = False
        m = None
        for pat in NAME_PATTERNS:
            m = pat.search(text)
            if m:
                break
        if m:
            name = m.group(1).strip().capitalize()
            if name.lower() not in NAME_BLOCKLIST and self.facts.get("name") != name:
                self.facts["name"] = name
                changed = True
        for key, pat, fmt in FACT_EXTRACTORS:
            fm = pat.search(text)
            if fm:
                val = fmt(fm)
                if self.facts.get(key) != val:
                    self.facts[key] = val
                    changed = True
        if changed:
            self._save_facts()

    def _save_facts(self) -> None:
        try:
            (self.mem_dir / "facts.json").write_text(
                json.dumps(self.facts, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def name(self) -> str | None:
        return self.facts.get("name")

    def relevant_fact(self, query: str) -> str | None:
        """A remembered fact whose words resurface in the query."""
        if not self.facts:
            return None
        qwords = {w for w in re.findall(r"[a-z]{3,}", query.lower())}
        for key in ("dream", "fear", "home", "work", "age"):
            val = self.facts.get(key)
            if not val:
                continue
            words = {w for w in re.findall(r"[a-z]{3,}", val)}
            if words & qwords:
                return f"You told me once that you {val} - and here it is again."
        return None

    def _refit(self) -> None:
        """Rebuild the memory TF-IDF index over remembered turns."""
        docs = [f"{t.get('user', '')} {t.get('aion', '')}" for t in self.turns]
        docs = [d for d in docs if d.strip()]
        if not docs:
            self._mem_vec = None
            self._mem_docs = []
            return
        try:
            self._mem_vec = TfidfVectorizer(
                ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=40000
            )
            self._mem_mat = self._mem_vec.fit_transform(docs)
            self._mem_docs = docs
        except Exception:
            self._mem_vec = None
            self._mem_docs = []

    # ------------------------------------------------------------------
    # recall
    # ------------------------------------------------------------------
    def relevant_memories(self, query: str, k: int = 3) -> list[str]:
        """Remembered exchanges relevant to the query, phrased as memory."""
        if self._mem_vec is None or not self.turns:
            return []
        try:
            import numpy as np
            q = self._mem_vec.transform([query])
            sims = cosine_similarity(q, self._mem_mat).ravel()
            idx = np.argsort(-sims)[:k]
            out = []
            for i in idx:
                if sims[int(i)] > 0.12:
                    t = self.turns[int(i)]
                    what = t.get("user", "")[:160]
                    out.append(f"You once told me: \"{what}\"")
            return out
        except Exception:
            return []

    def mood_profile(self) -> dict:
        topics = self.profile.get("topics", {})
        vibes = self.profile.get("vibes", {})
        fav = max(topics, key=topics.get) if topics else None
        vibe = max(vibes, key=vibes.get) if vibes else None
        return {
            "favorite_topic": fav,
            "vibe": vibe,
            "name": self.facts.get("name"),
            "n_user_turns": self.profile.get("n_user_turns", 0),
            "n_sessions": self.profile.get("n_user_turns", 0) // 20,
        }

    def stats(self) -> dict:
        topics = self.profile.get("topics", {})
        vibes = self.profile.get("vibes", {})
        return {
            "remembered_turns": len(self.turns),
            "topics_learned": topics,
            "vibes_learned": vibes,
            "facts_known": dict(self.facts),
            "memory_file": str(self.memory_file),
        }

    def forget(self) -> None:
        self.turns = []
        self.profile = {}
        self.facts = {}
        try:
            self.memory_file.unlink(missing_ok=True)
            self.profile_file.unlink(missing_ok=True)
            (self.mem_dir / "facts.json").unlink(missing_ok=True)
        except Exception:
            pass
        self._refit()
