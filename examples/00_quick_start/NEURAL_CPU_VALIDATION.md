# Neural quickstarts on CPU

Validated on 2026-11-22 using separate newly created Ubuntu 22.04 WSL /
Python 3.10.12 virtual environments. Both environments installed this fork
and passed `pip check`. All notebook code cells were executed, including
MovieLens 100K download/loading, training, prediction and evaluation.

## TensorFlow: NCF and RBM

From the repository root:

```bash
python3.10 -m venv .venv-tf
source .venv-tf/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[cpu-tensorflow]' -c examples/00_quick_start/constraints-tensorflow-cpu-py310.txt
python -m pip check
python tools/validate_quickstarts.py --cpu-only --output "$HOME/neural-tf-first-run" --notebooks ncf_movielens rbm_movielens
```

This profile uses `tensorflow-cpu==2.15.1`, Keras 2.15 and tf-slim 1.1.
These models use TensorFlow's v1 compatibility interfaces; TensorFlow 2.16+
is outside this tested profile. No CUDA runtime or cloud account is needed.

NCF trains for **5 epochs** by default. `NCF_EPOCHS=50` selects the original
training budget. A 50-epoch attempt was interrupted for the first-run budget
change and is not recorded as a completed validation. The validated run uses
the complete 100K dataset, produces unseen-item scores, checks probability
bounds/nonmissing predictions, and records finite ranking metrics and GPU count.

RBM trains for **30 epochs** with 600 hidden units. A syntax error in its version
print was corrected. Recommendations now use training history `Xtr`; held-out
`Xtst` supplies evaluation targets only. Assertions check finite outputs and
zero scores for already-seen training entries. The checkpoint is saved and
reloaded; sessions are closed before replacement and after use.
The reload check compares saved/restored trainable weights and runs both
`predict` and unseen-item recommendations on two users. `load()` now initializes
the static ratings lookup table after restoring weights. `remove_seen=True`
excludes nonzero observations in the supplied `x`, including after reload and
for a subset of users; callers should pass training history as `x`.

## PyTorch: FastAI

Use a separate environment so that TensorFlow and PyTorch dependency families
can be reproduced independently:

```bash
python3.10 -m venv .venv-fastai
source .venv-fastai/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[fastai]' -c examples/00_quick_start/constraints-fastai-cpu-py310.txt
python -m pip check
python tools/validate_quickstarts.py --cpu-only --output "$HOME/neural-fastai-first-run" --notebooks fastai_movielens
```

Validated FastAI 2.8.12 with `torch==2.5.1+cpu`, torchvision 0.20.1+cpu.
The notebook explicitly uses CPU data loaders and two PyTorch threads, trains
for **5 epochs**, exports/reloads the model, and compares predictions before
and after export. Runtime checks reported CUDA unavailable and zero GPUs.

Only load FastAI pickle exports that you created or trust.

## Results and remaining scope

| Notebook | Completed training | Result | Notebook seconds |
| --- | --- | --- | ---: |
| NCF | 5 epochs, full MovieLens 100K | Passed, GPU count 0, finite bounded predictions | 64.85 |
| FastAI | 5 epochs, full MovieLens 100K | Passed, CPU loader and export/reload prediction equality | 42.85 |
| RBM | 30 epochs, full MovieLens 100K | Passed, unseen-item checks and checkpoint reload | 16.12 |

The [machine-readable results](neural-validation-results.json) record source
hashes, execution times, metrics and the tested environment. Timings exclude
dependency installation and are not model speed comparisons. Data downloads
may use a warm cache. This validates executable first-use paths rather than
production recommendation quality; the three models use different evaluation
splits and training budgets, so their ranking metrics should not be compared
as a controlled model benchmark.

Existing NCF dataset, NCF model and RBM tests passed **25/25** on CPU with GPUs
hidden. TensorFlow compatibility/deprecation warnings remain; the fixed 2.15
profile passed. All checks ran locally; no confirmation Actions were started.

| Notebook group | GPU requirement and execution scope | Data/service requirement |
| --- | --- | --- |
| NCF, RBM, FastAI | CPU execution verified; no GPU required for these validated profiles | Public MovieLens 100K download |
| DKN, LSTUR, NAML, NPA, NRMS | [CPU synthetic execution verified](NEWS_CPU_VALIDATION.md), one epoch; actual MIND unverified | Default offline fixtures; trusted prepared MIND inputs optional |
| SASRec, xDeepFM, Wide & Deep | [CPU execution now validated](ADVANCED_CPU_VALIDATION.md); GPU not needed for those profiles | Actual MovieLens and explicitly synthetic data, as documented |
| Other sequential models | Not executed here; no CPU support guarantee established | Model-specific Amazon data and compatible TensorFlow APIs |
| AzureML / AzureML Designer notebooks | AzureML service access required for the workflow; GPU depends on selected compute | Azure account, workspace and compute configuration |

GPU builds, other Python versions, native Windows training, distributed
execution and cloud deployments remain outside this measured support scope.
NCF intermediate CSVs and RBM checkpoints use temporary directories instead of
leaving training files beside the source notebooks.
