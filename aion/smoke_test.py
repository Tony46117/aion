"""Smoke test: engine, humanizer, learner, voice wiring.

Run:  python -m aion.smoke_test
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from aion.engine import Engine
from aion.humanizer import humanize
from aion.voice import espeak_available


def main() -> int:
    tmp = Path(tempfile.mkdtemp()) / ".aion-test"
    eng = Engine()
    eng.learner = eng.learner.__class__(mem_dir=tmp)  # isolated memory for the test

    print("== talk mode ==")
    r1 = eng.reply("I keep having dreams about drowning. What do they mean?")
    print(r1[:300], "...\n")

    print("== deep mode ==")
    r2 = eng.reply("I am afraid of my own anger.", deep=True)
    print(r2[:300], "...\n")

    print("== learning check ==")
    assert len(eng.learner.turns) == 2, "learner must record every turn"
    assert (tmp / "memory.jsonl").exists(), "memory must persist"
    r3 = eng.reply("I told you before about the drowning dreams - they came back.")
    mem = eng.learner.relevant_memories("drowning dreams")
    print("recalled memories:", mem)
    assert mem, "learner must recall relevant memories"

    print("== humanizer standalone ==")
    h = humanize(
        "The dream is the small hidden door in the deepest and most intimate "
        "sanctum of the soul. It is not a disguise. It is a message.",
        rng=None,
    )
    print(h[:200], "...\n")
    assert h != h.title()  # sanity

    print("== stats ==")
    print(eng.stats())

    print("\nvoice available:", espeak_available())
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
