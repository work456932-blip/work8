# External data and evaluators

Prepare the released HEM annotations and feature caches locally:

| Dataset | Video features | Annotation / language interface |
|---|---|---|
| TACoS | C3D, 4096 channels | HEM JSON and GloVe text features |
| ActivityNet Captions | HEM C3D representation, 500 channels | HEM JSON and GloVe text features |
| MAD | Released CLIP-B/32, 512 channels | Converted released MAD JSON and released token-feature HDF5 |
| HI-ActivityNet | Same ActivityNet feature cache | Official released relevant/irrelevant queries; official strong-subset membership |

Dataset paths, temporal strides and windows are in `configs/`. Features,
annotations and pretrained models are external inputs. No dataset or model
is downloaded as part of configuration inspection or unit tests.

## Official HI-ActivityNet

`configs/hi_anet.yaml` trains against ordinary ActivityNet data if used for
training, and changes only evaluation to the official held-out benchmark.
The annotation adapter must return the existing HEM loader schema:

```json
{
  "benchmark": "HI-ActivityNet",
  "hi_test": {
    "VIDEO_ID": {
      "fps": 30.0,
      "num_frames": 3000,
      "num_clips": 374,
      "duration": 100.0,
      "annotations": [
        {
          "sentence_id": "UNIQUE_LOCAL_ID",
          "official_query_id": "RELEASED_ID",
          "sentence": "RELEASED_QUERY",
          "is_relevant": true,
          "is_strong": false,
          "segment": [10.0, 20.0]
        }
      ]
    }
  }
}
```

The numeric values above illustrate schema only. Preserve actual metadata,
query IDs, sentences, intervals, relevance and difficulty membership from the
release. An irrelevant pair has `is_relevant=false` and may use `[-1,-1]` for
its absent interval. Do not infer relevance from a generated cross-video pair.
Supply a release-specific converter through:

```bash
python tools/prepare_hi_annotations.py --source OFFICIAL_RELEASE \
  --adapter your_adapter:convert_annotations \
  --output data/hi_activitynet/annotations/official_hem_format.json
```

The converter is called with the source path and returns the dictionary above.
A preconverted official JSON can instead be validated without `--adapter`.

The exact official RA-IoU/F1 evaluator is an external dependency, with its
release-specific I/O isolated behind this interface:

```python
def evaluate(records, converted_annotation_path):
    # Translate records into the official release's prediction format.
    # Invoke the released evaluator and return its measured metrics dictionary.
    ...
```

Each record carries `official_query_id`, `video_id`, `is_relevant`, `is_strong`,
`segments`, `scores`, FP alternatives, `refused` and `cure_probability`.
Segment rows contain `[start_seconds, end_seconds, start_emission_seconds,
end_emission_seconds]`. Pass the first two values as the predicted interval;
use the appropriate FP or no-FP alternative for the checkpoint's protocol.
A refused record has empty candidate lists.

```bash
bash scripts/eval.sh configs/hi_anet.yaml CHECKPOINT \
  eval.official_evaluator=your_adapter:evaluate
```

The code exports predictions and requires the official adapter before
reporting HI metrics. It does not substitute a locally invented F1 score.

## Optional validation calibration

The paper fixes the selected cure threshold at 0.5. To calibrate a new run,
use a validation split disjoint from reported evaluation, never HI-ActivityNet:

```bash
python tools/augment_validation.py \
  --input data/anet_1.3/annotations/anet_1.3.json --split val_1 \
  --output data/anet_1.3/annotations/validation_augmented.json
bash scripts/eval.sh configs/anet.yaml CHECKPOINT name=validation_predictions \
  eval.data.anno_file=data/anet_1.3/annotations/validation_augmented.json \
  eval.data.split=val_1 eval.explicit_refusal=false
python tools/calibrate_refusal.py \
  --predictions results/validation_predictions/predictions.json \
  --output calibration/anet.json
```

Then pass `eval.calibration_file=calibration/anet.json` for subsequent evaluation.
Direct-MT needs its own calibration file. The ordinary validation augmentation
is explicitly separate from the verified official benchmark.
