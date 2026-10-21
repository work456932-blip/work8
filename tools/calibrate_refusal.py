#!/usr/bin/env python3
"""Freeze a cure threshold and equal-width ECE bins using validation only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.calibration import select_calibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bins', type=int, default=15)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error('--bins must be at least 2')
    value = select_calibration(json.loads(args.predictions.read_text()), args.bins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
