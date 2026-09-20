# aion — a Jungian AI for the terminal

> "One does not become enlightened by imagining figures of light, but by making the darkness conscious."

**aion** is a terminal CLI chatbot that speaks as a representation of C.G. Jung. It is
trained (small-scale, on-device) on a corpus of Jungian books, runs fully offline in
pure Python, and grows from every conversation you have with it.

## Architecture

| Module | Purpose |
|---|---|
| `cli.py` | Terminal app (Rich + prompt_toolkit) — the opencode-style UI |
| `engine.py` | Jung persona brain: TF-IDF retrieval + char-ngram logistic classifier + generative composer |
| `humanizer.py` | Humanizes engine output: contractions, hedging, rhythm, Jungian interjections |
| `learner.py` | Learns from every conversation; persists memory to `~/.aion/` so aion gets smarter over time |
| `voice.py` | Jung's voice: espeak-ng TTS with a lowered pitch/slowed cadence (bilingual German accent voice) |
| `train.py` | Trains the persona on the corpus (TF-IDF + logistic head on teachings, char-CNN on style), 100+ epochs |
| `ingest.py` | Extracts and cleans the raw text of the PDF books into `data/corpus.txt` |
| `train_cli.py` | Thin wrapper to run training from the installed `aion` command |

Trained artifacts live in `models/` (committed) and are loaded by the CLI at startup.
Long-term memory (what aion learns from you) lives outside the repo in `~/.aion/`.

## Corpus

The persona is trained on the Jungian books in `aion/data/books/`:

- *Man and His Symbols* — C.G. Jung (with M.-L. von Franz, J. Henderson, J. Jacobi, A. Jaffé)
- *Catafalque: Carl Jung and the End of Humanity* — Peter Kingsley

Run `python -m aion.ingest` to rebuild `data/corpus.txt` whenever a book is added,
then retrain (see below).

## Install

```bash
python3.12 -m venv python312
source python312/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires the `espeak-ng` system package for `/voice` (`sudo apt-get install espeak-ng`).

## Train

```bash
python train.py --epochs 120            # full training, 100+ epochs
python train.py --epochs 10 --quick     # quick smoke training
```

Each epoch trains:
1. a TF-IDF + LogisticRegression "teachings" classifier over the corpus (retrieval head),
2. a char-CNN (PyTorch) over Jung's own sentence style (generative style head).

Artifacts are written to `models/` with a training report in `models/training_report.json`.

## Run

```bash
aion
```

Then, inside the session:

| Command | Action |
|---|---|
| `/chat` | opens the chat menu → choose **talk** (conversation) or **deep** (deep analysis session) |
| `/voice` | toggles Jung's voice — spoken answers with his cadence |
| `/help` | list all commands |
| `/stats` | what aion has learned from your conversations so far |
| `/forget` | wipe learned memory |
| `/quit` | leave (aion remembers your conversations) |

## Persona

aion answers as Jung himself would: first-person, measured, symbolic, pointing always
back to the inner life — the shadow, the anima/animus, complexes, dreams, individuation.
The `humanizer` module keeps the voice imperfect and alive: contractions, hedges,
rhetorical questions, the occasional German word — the way a person actually talks,
not the way a machine writes.

---

*"The meeting of two personalities is like the contact of two chemical substances:
if there is any reaction, both are transformed."* — C.G. Jung
