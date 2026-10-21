#!/usr/bin/env python3
"""Validate a lossless HEM-format conversion of OFFICIAL HI-ActivityNet.

The release-specific conversion is deliberately supplied as an adapter. No
cross-video query generation or label inference is performed by this tool.
"""
import argparse
import importlib
import json
from pathlib import Path


def validate_official_annotations(data):
    if data.get('benchmark') != 'HI-ActivityNet':
        raise ValueError('A converted official benchmark marker is required')
    splits = [k for k, value in data.items() if isinstance(value, dict)]
    if any(k.lower().startswith(('train', 'val', 'dev')) for k in splits):
        raise ValueError('HI-ActivityNet is evaluation-only in this protocol')
    ids = set()
    for split in splits:
        for video in data[split].values():
            for pair in video['annotations']:
                for key in ('sentence', 'is_relevant', 'official_query_id', 'is_strong'):
                    if key not in pair:
                        raise ValueError('The adapter must preserve released ' + key)
                if not isinstance(pair['is_relevant'], bool) or not isinstance(pair['is_strong'], bool):
                    raise ValueError('Preserve boolean relevance and released strong-subset labels')
                if pair['official_query_id'] in ids:
                    raise ValueError('Duplicate official query ID')
                ids.add(pair['official_query_id'])
                if pair['is_relevant'] and ('segment' not in pair or len(pair['segment']) != 2):
                    raise ValueError('Relevant queries require the original interval')
    if not ids:
        raise ValueError('No official queries supplied')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', help='module:function; called with source path; returns HEM-format dict')
    args = parser.parse_args()
    if args.adapter:
        module, function = args.adapter.split(':')
        data = getattr(importlib.import_module(module), function)(str(args.source))
    else:
        data = json.loads(args.source.read_text())
    data = validate_official_annotations(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))


if __name__ == '__main__':
    main()
