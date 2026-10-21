from pathlib import Path

from omegaconf import OmegaConf


def test_all_configs_merge_with_survival_defaults():
    root = Path(__file__).resolve().parents[1]
    default = OmegaConf.load(root / 'configs/default.yaml')
    for name in ('anet', 'tacos', 'mad', 'hi_anet'):
        merged = OmegaConf.merge(default, OmegaConf.load(root / f'configs/{name}.yaml'))
        assert merged.model.survival_head.embd_dim == merged.model.vid_net.embd_dim
        assert merged.train.amp_dtype == 'bfloat16'
        assert merged.train.survival.enabled


def test_variants_keep_matched_decoder_config_and_hem_has_no_head():
    from libs.config import load_config
    root = Path(__file__).resolve().parents[1]
    for dataset in ('anet', 'tacos', 'mad'):
        full = load_config(dict(config=str(root / f'configs/{dataset}.yaml')))
        direct = load_config(dict(config=str(root / f'configs/{dataset}.yaml'), variant='direct_mt'))
        hem = load_config(dict(config=str(root / f'configs/{dataset}.yaml'), variant='hem'))
        assert full.model.survival_head == direct.model.survival_head
        assert direct.train.survival.boundary_objective == 'direct'
        assert hem.model.survival_head is None
        assert hem.train.data.prefix_sampling_prob == 0
        assert hem.train.data.hard_negative_prob == 0
        assert full.train.data.hard_negative_mode == 'lexical'
        assert full.eval.refusal_threshold == .5
