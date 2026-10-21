#!/usr/bin/env python3
"""Aggregate metrics.json files from repeated seeds into mean/std JSON."""

import argparse
import json
from pathlib import Path

import numpy as np


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, path))
        elif isinstance(item, (int, float)):
            result[path] = float(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('metrics', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    runs = []
    for path in args.metrics:
        with path.open() as handle:
            runs.append(flatten(json.load(handle)))
    keys = sorted(set.intersection(*(set(run) for run in runs)))
    summary = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[key] = {
            'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std(ddof=1)) if len(finite) > 1 else None,
            'values': [float(x) if np.isfinite(x) else None for x in values],
            'num_defined_runs': int(len(finite)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print('wrote', args.output)


if __name__ == '__main__':
    main()
