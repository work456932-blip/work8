#!/usr/bin/env python3
"""Convert official MAD annotations to the grouped OnVTG/SurvVTG schema."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-dir', required=True, type=Path)
    parser.add_argument('--video-features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fps', type=float, default=5.0)
    args = parser.parse_args()

    split_files = {
        'train': args.annotation_dir / 'MAD_train.json',
        'val': args.annotation_dir / 'MAD_val.json',
        'test': args.annotation_dir / 'MAD_test.json',
    }
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError('missing MAD annotations: ' + ', '.join(missing))

    with h5py.File(args.video_features, 'r') as features:
        feature_lengths = {movie: int(features[movie].shape[0]) for movie in features.keys()}

    converted = {}
    for split, annotation_file in split_files.items():
        with annotation_file.open() as handle:
            records = json.load(handle)
        grouped = defaultdict(list)
        for sentence_id, item in records.items():
            movie = item['movie']
            grouped[movie].append({
                'sentence': item['sentence'].strip(),
                'sentence_id': sentence_id,
                'segment': [float(x) for x in item['ext_timestamps']],
                'is_relevant': True,
            })

        split_output = {}
        for movie, annotations in grouped.items():
            if movie not in feature_lengths:
                continue
            num_clips = feature_lengths[movie]
            duration = max(
                float(records[annotation['sentence_id']]['movie_duration'])
                for annotation in annotations
            )
            split_output[movie] = {
                'fps': args.fps,
                'num_frames': int(round(duration * args.fps)),
                'num_clips': num_clips,
                'duration': duration,
                'annotations': annotations,
            }
        converted[split] = split_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(converted, handle)
    print('wrote', args.output, {key: len(value) for key, value in converted.items()})


if __name__ == '__main__':
    main()
