# Wide & Deep, SASRec and xDeepFM CPU validation

Validated on 2026-11-22 in a newly created Ubuntu 22.04 WSL / Python 3.10.12
virtual environment with TensorFlow CPU 2.15.1, Keras 2.15.0 and zero visible
GPUs. The environment was installed from this fork and passed `pip check`.

## Installation and execution

From the repository root:

```bash
python3.10 -m venv .venv-advanced
source .venv-advanced/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[cpu-tensorflow]' -c examples/00_quick_start/constraints-tensorflow-cpu-py310.txt
python tools/validate_quickstarts.py --cpu-only --output "$HOME/advanced-first-run" --notebooks wide_deep_movielens sasrec_amazon xdeepfm_criteo
```

The legacy notebook filenames remain for existing links. `sasrec_amazon.ipynb`
now defaults to the SASREC model and MovieLens sequences; its former default
selected SSEPT. `xdeepfm_criteo.ipynb` defaults to labeled synthetic FFM data.
Each notebook runs in its own kernel, keeping SASRec eager execution separate
from the TensorFlow v1-compatible Estimator/DeepRec models.

## Measured results

| Model | Data | Training budget | Result | Notebook seconds |
| --- | --- | --- | --- | ---: |
| Wide & Deep | Actual MovieLens 100K including genres | 500 steps, batch 512 | Passed; exported model independently loaded and predicted | 49.59 |
| SASRec | Actual MovieLens 100K chronological sequences | 5 epochs | Passed; 943 users / 1,682 items | 30.95 |
| SASRec offline mode | Synthetic sequences | 5 epochs | Passed; 32 users / 120 items | 8.73 |
| xDeepFM | Synthetic clicks: 1,600 train / 200 validation / 200 test, 39 fields | 10 epochs | Passed; AUC 0.8998, logloss 0.3609 | 11.48 |

[JSON results](advanced-validation-results.json) include metrics, LF-normalized
source hashes and data labels. Notebook times include kernel execution and
exclude installation; caches and concurrent local work affect timings. They
are not speed comparisons. Actual MovieLens results and synthetic results
describe their respective datasets, not production recommendation quality.

SASRec NDCG@10 and Hit@10 use the library's sampled ranking evaluation: one
positive and 100 sampled negative candidates, rather than full-catalog ranking.
The inherited evaluator samples with replacement and excludes training history;
these values are not a controlled benchmark against the other models.

## Repaired failures and options

- Wide & Deep indexed the first GPU in an empty list and failed on CPU. GPU
  information is now optional. The default CPU budget is 500 steps and metrics
  run after training; `WIDE_DEEP_STEPS=50000` selects the historical budget,
  which was not validated here.
- Set `WIDE_DEEP_EXPORT_DIR` to retain a model. Default export checks use a
  temporary directory. Persistent exports return a decoded, usable path;
  `tf.saved_model.load(path, tags=["serve"])` loads the prediction graph. It
  accepts serialized `tf.train.Example` values with `userID`, `itemID`, and
  multi-hot `genre` features using the notebook's genre encoding. An independent
  process loaded the retained export and produced a finite prediction. The
  verification export was deleted after this check.
- SASRec now passes the actual `learning_rate` argument, closes sampler workers
  in `finally`, and uses temporary sequence files. Local inputs that have no
  trainable user or no unseen catalog item fail before workers start. Both
  short-history and saturated-history rejection cases were executed.
- The historical xDeepFM resource host fails DNS resolution. The synthetic
  profile defines hyperparameters locally and generates seeded, disjoint FFM
  splits. No real Criteo data was executed or redistributed. For local prepared
  files set `XDEEPFM_DATA_SOURCE=local`, `XDEEPFM_DATA_DIR` containing `train.ffm`,
  `valid.ffm`, `test.ffm`, and `XDEEPFM_FEATURE_COUNT`; field count is 39 and
  feature IDs must use the same mapping across all splits. This local real-data
  option was not validated with Criteo data.
- All three notebooks now select a portable `python3` kernel instead of a
  removed GPU environment. Use the kernel installed in the validated environment.

For SASRec offline mode, set `SASREC_DATA_SOURCE=synthetic`. For prepared local
chronological sequences set `SASREC_DATA_SOURCE=local` and
`SASREC_SEQUENCE_PATH` to a two-column tab-separated file with contiguous
one-based user and item IDs. Amazon data was not downloaded or validated.

## Regression checks and cleanup

Thirty existing Wide & Deep, SASRec/SSEPT-construction, NCF and RBM tests passed
in this new environment. Two malformed local sequence inputs were rejected
without starting indefinite sampling. Independent review findings about data
labels, worker guards, export retention and kernel metadata were corrected.

Generated sequences, FFM data and temporary checkpoints are cleaned at notebook
completion. After preserving validation receipts, the superseded TensorFlow
environment and this trial's retained export, obsolete wheel and test input
files were removed (about 2.28 GB of file bytes). The final validation environment
and repository source remain available. No shared package caches or user datasets
were deleted. No confirmation GitHub Actions were run.

This is CPU first-run validation for these profiles. Longer training budgets,
GPU/distributed execution, native Windows execution, actual Amazon/Criteo
benchmarks and cloud deployments remain outside the measured scope.
