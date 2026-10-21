#!/usr/bin/env python3
"""Re-evaluate actual observed prefixes (including Direct-MT softmax) by state."""
import json
import math
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.config import load_config
from libs.evaluator import Evaluator
from libs.survival_metrics import compute_survival_metrics
from omegaconf import OmegaConf


def main():
    cfg = load_config()
    cfg._distributed = False
    cfg.eval.explicit_refusal = False
    cfg.eval.save_curves = True
    evaluator = Evaluator(OmegaConf.to_object(cfg))
    evaluator.load(cfg.eval.ckpt)
    evaluator.model.eval()
    regimes = {name: [] for name in ('pre', 'active', 'post', 'full')}
    for batch in evaluator.dataloader:
        sample = batch[0]
        length = sample['vid'][0].size(-1)
        for q, target in enumerate(sample['target'].squeeze(1)):
            if not bool(sample['is_relevant'][q]):
                continue
            start, end = [float(x) for x in target]
            # Deterministic regime midpoints are an implementation default;
            # skip unavailable regimes rather than fabricating observations.
            endpoints = dict(pre=math.ceil(start/2), active=math.ceil((start+end)/2)+1,
                             post=math.ceil((end+length-1)/2)+1, full=length)
            for mode, stop in endpoints.items():
                t = stop-1
                if not 1 <= stop <= length:
                    continue
                if mode == 'pre' and not t < start: continue
                if mode == 'active' and not start <= t < end: continue
                if mode == 'post' and not end <= t: continue
                prefix = deepcopy(sample)
                prefix['vid'] = (sample['vid'][0][..., :stop],)
                prediction = evaluator.predict(prefix)[q]
                regimes[mode].append(prediction)
    summary = {name: compute_survival_metrics(values) for name, values in regimes.items()}
    out = Path(cfg._root) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'prefix_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == '__main__':
    main()
