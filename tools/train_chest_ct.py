"""Train the optional research classifier using the separately downloaded CT archive."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from healthdl.chest_training import train_chest
from healthdl.data import ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/external/chest_ct/sarscov2-ctscan-dataset.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/chest_ct")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    train_chest(args.archive, args.output, seed=args.seed, epochs=args.epochs)
