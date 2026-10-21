# Implementation protocol

The implementation follows the supplied main paper (Sections 3 and 4) and
Supplementary A / Table S6. It extends the supplied HEM-based code; it does not
contain trained weights, datasets, or a new measurement of paper results.

| Paper item | Implementation |
|---|---|
| Unchanged HEM proposal/memory trunk | `libs/modeling/{video_net,text_net,fusion,head}.py`; HEM losses remain in `libs/trainer.py` |
| Finest query-conditioned feature | `OnVTG.fuse_and_predict`, FPN level 0 |
| Causal decoder | Width 256, two kernel-3 blocks, dilations 1/2, left padding, channel-only normalization, GELU, dropout |
| Four separate heads | Start/end/cure scalars and three state logits; kernel-1 convolutions are pointwise linear maps |
| Right/left censoring | `libs/modeling/survival_loss.py`; LC starts excluded; a crop already active enters end risk at its origin; boundaries entirely before crop are excluded |
| End risk | Relevant, started prefixes only; irrelevant/pre-event samples never contribute end hazard likelihood |
| Composite objective | HEM localization + optional HEM FP + start/end + 0.5 cure + 0.5 state + 0.05 monotonic |
| Prefix augmentation | 0.70; pre/active/post weights 0.35/0.35/0.30; full annotation remains unchanged |
| Training negatives | Cross-video lexical overlap, 0.25 on TACoS/ActivityNet, 0.20 on MAD; augmentation labels are not verified benchmark labels |
| Direct-MT | Identical decoder and trainable parameter shapes; softmax over observed positions plus out-of-prefix; future/irrelevant boundaries use out-of-prefix; LC boundaries masked; same cure/state/monotonic losses |
| HEM control | `variant=hem` removes the added decoder and augmentation and retains original localization / optional FP objectives |
| Inference | Proposal score times (1-cure) at that proposal's causal emission endpoint; state probabilities remain diagnostic |
| Refusal | Paper's validation-selected threshold 0.5 (main Sec. 4.6); HI annotations are never used to select it |
| Reporting | Seeds 2024/2025/2026; arithmetic mean and sample standard deviation; fixed diagnostic seed 2024 |
| HI-ActivityNet | Official released queries and strong-subset labels only; held out from training/calibration; exact official evaluation supplied through an adapter |
| Efficiency | Single A100 40GB, BF16/TF32, identical streams/windows/warm-up and synchronized timing |

## Explicit defaults for details not fixed in the paper

These are implementation choices, not recovered experimental records:

- Keep supplied HEM dataset windows, feature strides, optimizer, learning rates,
  warm-up and epoch schedules. `epochs` means post-warm-up epochs; total training
  is `epochs + warmup_epochs` (17 for TACoS/ActivityNet; 10 for MAD).
- Keep supplied dropout 0.1, start/end prior 0.01 and cure prior 0.10.
- Keep supplied risk-length normalization for event NLL; the numerator is the
  exact observed/censored log likelihood. Set
  `train.survival.normalize_by_risk=false` for the unreduced per-query likelihood.
  Evaluation uses the same reduction. Joint NLL adds valid start/conditional-end
  contributions before averaging examples, without cure BCE.
- Discretize fractional grid boundaries by `ceil`: an event first becomes
  visible at the first observed grid location at or beyond its boundary.
- Direct-MT's additional class has a fixed zero reference logit. This gives a
  categorical boundary distribution without adding parameters to the matched
  decoder. Its distributions are evaluated as categorical masses, not sigmoid
  hazards. Conditional end diagnostics restrict and renormalize the end mass to
  the valid risk set; training retains Supp. Eq. 2's direct categorical targets.
- Use one randomly selected relevant query as the prefix-regime anchor when
  multiple queries share a crop. Renormalize regime weights over feasible
  regimes; never relabel a future event as a negative.
- Lexical matching uses a compact inverted index, excludes high-frequency
  postings, and samples among the highest-overlap cross-video queries. When
  no overlapping candidate exists, retain the relevant query.
- ECE uses 15 equal-width probability bins, fixed before test. Start ECE/Brier
  use the prefix-conditioned mixture survival; end scores use survival
  conditional on risk entry. IBS averages temporal squared errors within an
  example, then examples; `ibs` averages its available start/end terms.
- Optional fresh threshold calibration maximizes never-event detection F1 on
  ordinary validation pairs augmented with lexical cross-video queries; ties
  favor the larger threshold. The published operating point stays 0.5 by default.
- Default scripts use the last checkpoint, avoiding selection on the reported
  test split. Checkpoint-selection rules are not specified by the paper. For
  early stopping, explicitly supply a separate validation split.
- Deterministic pre/active/post midpoint prefixes are available for diagnostic
  evaluation. The paper does not fix these endpoint positions. Direct-MT's
  softmax is recomputed for every evaluated prefix.

## Entry points

```bash
# Same dataset/backbone configuration, three model variants:
bash scripts/train.sh configs/anet.yaml variant=survvtg
bash scripts/train.sh configs/anet.yaml variant=direct_mt
bash scripts/train.sh configs/anet.yaml variant=hem

# Disable the original future-prediction branch and loss together:
bash scripts/train.sh configs/anet.yaml variant=no_fp

# Paper ablation definitions, fixed seed, no FP:
bash scripts/run_ablations.sh anet 2024

# Three seeds (change anet to tacos or mad):
bash scripts/run_seeds.sh anet 2024 2025 2026

# Prefix diagnostics (must match checkpoint's model/FP configuration):
python tools/evaluate_prefixes.py config=configs/anet.yaml eval.ckpt=CHECKPOINT
```

The no-gate ablation re-evaluates the full checkpoint. It does not retrain or
change survival predictions. Head-removal ablations physically omit the
corresponding heads and return undefined metrics for them.
