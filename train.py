"""Run with `python train.py --help`; all paths are independent of the shell cwd."""

import argparse
from pathlib import Path

from healthdl.data import ROOT
from healthdl.training import TrainConfig, train


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the symptom neural network on CPU")
    parser.add_argument("--data", type=Path, default=ROOT / "Training.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--no-augmentation", action="store_true", help="Train the original-pattern baseline without synthetic variants")
    args = parser.parse_args()
    report = train(TrainConfig(data_path=args.data, output_dir=args.output,
                               seed=args.seed, epochs=args.epochs, patience=args.patience,
                               augment_missing_symptoms=not args.no_augmentation))
    print(f"Unique patterns: {report['unique_rows']}; partitions: {report['split_sizes']}")
    print(f"Selected epoch: {report['best_epoch']}; held-out accuracy: {report['test']['accuracy']:.3f}; "
          f"macro F1: {report['test']['macro_f1']:.3f}")
    print(f"Saved artifacts to {args.output.resolve()}. Dataset results are not clinical validation.")


if __name__ == "__main__":
    main()
