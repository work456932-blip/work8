#!/usr/bin/env python3
"""Measure parameter count, A100 memory, and online forward latency."""

import argparse
import json
from pathlib import Path
import time

from omegaconf import OmegaConf
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.modeling import OnVTG
from libs.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('configs/mad.yaml'))
    parser.add_argument('--variant', choices=('survvtg', 'direct_mt', 'hem'), default='survvtg')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--frames-per-step', type=int, default=1, help='Explicit frame accounting for FPS; one encoded stream step by default')
    parser.add_argument('--features', type=Path, help='Optional shared [T,C] .npy stream; synthetic features otherwise')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--video-length', type=int, default=1024)
    parser.add_argument('--text-length', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('benchmark.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for the benchmark')

    cfg = load_config(dict(config=str(args.config), variant=args.variant))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    model = OnVTG(OmegaConf.to_object(cfg.model)).cuda().eval()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint.get('model_ema', checkpoint), strict=True)

    video = torch.randn(
        1, 1, cfg.model.vid_net.in_dim, args.video_length,
        device='cuda', dtype=torch.float32,
    )
    if args.features:
        import numpy as np
        features = np.load(args.features)[:args.video_length]
        if features.shape != (args.video_length, cfg.model.vid_net.in_dim):
            raise ValueError('Shared stream has incorrect length or channel dimension')
        video = torch.as_tensor(features.T.copy(), device='cuda', dtype=torch.float32)[None, None]
    video_mask = torch.ones(1, 1, args.video_length, device='cuda', dtype=torch.bool)
    text_length = min(args.text_length, cfg.model.text_net.max_seq_len)
    text = torch.randn(
        1, cfg.model.text_net.in_dim, text_length,
        device='cuda', dtype=torch.float32,
    )
    text_mask = torch.ones(1, text_length, device='cuda', dtype=torch.bool)
    text_size = torch.ones(1, device='cuda', dtype=torch.long)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(args.warmup):
            model(video, video_mask, text, text_mask, text_size)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(video, video_mask, text, text_mask, text_size)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_params = sum(parameter.numel() for parameter in model.parameters())
    survival_params = sum(
        parameter.numel() for parameter in model.survival_head.parameters()
    ) if model.enable_survival else 0
    result = {
        'device': torch.cuda.get_device_name(0),
        'variant': args.variant,
        'stream_source': 'external' if args.features else 'synthetic',
        'warmup': args.warmup,
        'seed': args.seed,
        'batch_size': 1,
        'frames_per_step': args.frames_per_step,
        'fps': args.iterations * args.frames_per_step / elapsed,
        'timing_scope': 'full observed-window forward; not an incremental cached-step benchmark',
        'video_length': args.video_length,
        'iterations': args.iterations,
        'latency_ms_mean': elapsed * 1000.0 / args.iterations,
        'throughput_prefixes_per_second': args.iterations / elapsed,
        'peak_memory_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'parameters_total': total_params,
        'parameters_survival': survival_params,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
