#!/usr/bin/env python3
"""Lexical cross-video validation negatives; these are NOT HI-ActivityNet."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import re


def augment_validation(source, split='val_1', seed=2024):
    if source.get('benchmark') == 'HI-ActivityNet' or not split.startswith(('val', 'dev')):
        raise ValueError('Use ordinary validation annotations only')
    rng = random.Random(seed)
    result = deepcopy(source[split])
    index = defaultdict(list)
    entries = []
    def tokens(text):
        return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    for vid, video in source[split].items():
        for pair in video['annotations']:
            entries.append((vid, pair))
            for token in tokens(pair['sentence']):
                index[token].append(len(entries)-1)
    for vid, video in result.items():
        additions = []
        for number, pair in enumerate(video['annotations']):
            candidates = set()
            for token in tokens(pair['sentence']):
                candidates.update(index[token])
            candidates = sorted(i for i in candidates if entries[i][0] != vid)
            if not candidates:
                continue
            _, chosen = entries[rng.choice(candidates)]
            additions.append(dict(sentence=chosen['sentence'], segment=[-1., -1.],
                                  is_relevant=False, sentence_id=f'{vid}_augneg_{number:06d}'))
        video['annotations'].extend(additions)
    return dict(benchmark='ActivityNet-validation-augmentation', **{split:result})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='val_1')
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    data = augment_validation(json.loads(args.input.read_text()), args.split, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))


if __name__ == '__main__':
    main()
