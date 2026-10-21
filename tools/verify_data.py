#!/usr/bin/env python3
"""Fail-fast verification of annotations and pre-extracted features."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def feature_keys(path):
    if path.suffix == '.h5':
        with h5py.File(path, 'r') as handle:
            return set(handle.keys())
    return {item.stem for item in path.glob('*.npy')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--video-features', required=True, type=Path)
    parser.add_argument('--text-features', type=Path)
    parser.add_argument('--splits', nargs='+', default=['train'])
    parser.add_argument('--expected-video-dim', type=int)
    parser.add_argument('--sample-count', type=int, default=8)
    args = parser.parse_args()

    with args.annotations.open() as handle:
        annotations = json.load(handle)
    missing_splits = [split for split in args.splits if split not in annotations]
    if missing_splits:
        raise KeyError(f'missing annotation splits: {missing_splits}')

    keys = feature_keys(args.video_features)
    required_videos = {
        video_id for split in args.splits for video_id in annotations[split]
    }
    missing_videos = sorted(required_videos - keys)
    if missing_videos:
        raise FileNotFoundError(
            f'{len(missing_videos)} videos have no feature; first={missing_videos[:5]}'
        )

    if args.text_features:
        text_keys = feature_keys(args.text_features)
        required_text = {
            annotation.get('sentence_id', f'{video_id}_{idx:04d}')
            for split in args.splits
            for video_id, video in annotations[split].items()
            for idx, annotation in enumerate(video['annotations'])
        }
        missing_text = sorted(required_text - text_keys)
        if missing_text:
            raise FileNotFoundError(
                f'{len(missing_text)} queries have no feature; first={missing_text[:5]}'
            )

    if args.expected_video_dim:
        sampled = sorted(required_videos)[:args.sample_count]
        if args.video_features.suffix == '.h5':
            with h5py.File(args.video_features, 'r') as handle:
                dims = {video: int(handle[video].shape[-1]) for video in sampled}
        else:
            dims = {
                video: int(np.load(args.video_features / f'{video}.npy', mmap_mode='r').shape[-1])
                for video in sampled
            }
        wrong = {video: dim for video, dim in dims.items() if dim != args.expected_video_dim}
        if wrong:
            raise ValueError(f'unexpected video feature dimensions: {wrong}')

    print(
        f'OK: {len(required_videos)} videos, {sum(len(annotations[s]) for s in args.splits)} split entries'
    )


if __name__ == '__main__':
    main()
