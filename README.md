# SurvVTG

Anonymous implementation of **Event-State Survival Modeling for Online Video
Temporal Grounding**. It extends HEM with causal start/end hazards, dynamic
never-event prediction, event states and censoring-aware supervision. Matched
HEM and Direct-MT controls are included.

Use `environment.yml` for the PyTorch/CUDA environment, and prepare the released
features and annotations at the paths in `configs/`.

```bash
bash scripts/train.sh configs/anet.yaml
bash scripts/eval.sh configs/anet.yaml results/anet/models/last.pth
```

Replace `anet` with `tacos` or `mad`. Select the matched controls with
`variant=hem` or `variant=direct_mt`. Optional HEM checkpoint initialization uses
`_pretrained=PATH`.

[Protocol and implementation defaults](docs/PROTOCOL.md) ·
[Data and official evaluation interfaces](docs/DATA.md)

Datasets, pretrained weights and the official HI-ActivityNet evaluator are
external inputs. This release does not claim an end-to-end reproduction of the
paper's numerical results. Third-party attribution is in
[THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md).
