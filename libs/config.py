"""Shared default -> dataset -> variant -> command-line resolution."""
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]

def load_config(cli=None):
    args = OmegaConf.from_cli() if cli is None else OmegaConf.create(cli)
    dataset = OmegaConf.load(args.config)
    del args.config
    variant = args.pop('variant', None)
    overlay = OmegaConf.load(ROOT / 'configs' / 'variants' / (str(variant) + '.yaml')) if variant else OmegaConf.create({})
    cfg = OmegaConf.merge(OmegaConf.load(ROOT / 'configs/default.yaml'), dataset, overlay, args)
    if cfg.model.variant == 'hem' and cfg.model.survival_head is not None:
        raise ValueError('HEM must not instantiate the survival decoder')
    expected = 'direct' if cfg.model.variant == 'direct_mt' else 'hazard'
    if cfg.model.variant != 'hem' and cfg.train.survival.boundary_objective != expected:
        raise ValueError('Model variant and boundary objective disagree')
    return cfg
