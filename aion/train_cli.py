"""Allow `aion-train` console entry point to run training."""
from .train import main

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="train the aion persona")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(epochs=a.epochs, quick=a.quick)
