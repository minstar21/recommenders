# News recommendation: CPU first run

This fork provides an offline first-run path for DKN, LSTUR, NAML, NPA and NRMS.
The default inputs are **synthetic**. Their metrics check execution, not MIND
recommendation quality. The architectures run with smaller dimensions and a
one-epoch budget. DKN retains both entity and context channels; their embeddings
are random, including files whose legacy names end in `100.npy` (here: 16 dimensions).

## Installation and execution

Validated on Ubuntu 22.04 under WSL, Python 3.10, tensorflow-cpu 2.15.1 and Keras
2.15.0, with no visible GPU. Start from a clone of this fork, not the upstream
PyPI release, which does not contain this helper or these notebook changes:

```bash
python3.10 -m venv .venv-news
source .venv-news/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[cpu-tensorflow]' -c examples/00_quick_start/constraints-tensorflow-cpu-py310.txt
python -m pip check
python tools/validate_quickstarts.py --cpu-only --notebooks dkn_MIND lstur_MIND naml_MIND npa_MIND nrms_MIND --output "$HOME/news-first-run"
```

The output directory must be new. The runner executes every cell in separate
notebook kernels, checks finite recorded metrics and writes executed copies
and `results.json`. `NEWS_EPOCHS=5` changes the training budget; only one epoch
was validated here. Interactive notebook use must select this environment's kernel.

## Actual MIND and prepared input support

On 2026-11-22, the original `recodatasets.z20.web.core.windows.net` resource
host failed DNS resolution. The alternative `recodatasets.blob.core.windows.net`
host also failed resolution. The documented official
`https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip`
returned HTTP 409. These are observed access failures in this environment.
[Upstream reports the same public-access restriction](https://github.com/recommenders-team/recommenders/issues/2133).
No real MIND training run was completed, and no unverified mirror was substituted.

If you already have a trusted compatible prepared dataset, set the following
variables before launching the same runner. The notebooks do not download
missing files or change prepared input files:

```bash
export MIND_DATA_DIR=/absolute/prepared-mind
export DKN_DATA_DIR=/absolute/prepared-dkn
```

| Models | Required prepared layout |
|---|---|
| LSTUR, NPA, NRMS | `train/{news,behaviors}.tsv`, `valid/{news,behaviors}.tsv`, `utils/embedding.npy`, `utils/word_dict.pkl`, `utils/uid2index.pkl`, `utils/{lstur,npa,nrms}.yaml` for each selected model |
| NAML | Same TSVs; `utils/embedding_all.npy`, `word_dict_all.pkl`, `uid2index.pkl`, `vert_dict.pkl`, `subvert_dict.pkl`, `naml.yaml` |
| DKN | `dkn.yaml`, `{train,valid,test}_mind_demo.txt`, `doc_feature.txt`, `user_history.txt`, `word_embeddings_100.npy`, `TransE_entity2vec_100.npy`, `TransE_context2vec_100.npy` |

Prepared YAML supplies model dimensions; notebooks override epochs and batch
size (8), plus logging in NRMS. Preserve dictionary-to-embedding index alignment.
Prepared input provenance is recorded as `local_prepared_unverified_provenance`:
a directory name does not establish that its contents are real MIND. Dictionaries
are Python pickles; load only dictionaries you created or trust. Raw MIND TSVs
alone are insufficient: these models also need compatible dictionaries,
embeddings, YAML and, for DKN, indexed entity/context features.

## Recorded results

| Model | Input | Train/evaluate/restore | Notebook seconds |
|---|---|---|---:|
| DKN | Synthetic, one epoch | Passed; GPU count 0 | 7.03 |
| LSTUR | Synthetic, one epoch | Passed; GPU count 0 | 6.62 |
| NAML | Synthetic, one epoch | Passed; GPU count 0 | 6.61 |
| NPA | Synthetic, one epoch | Passed; GPU count 0 | 5.71 |
| NRMS | Synthetic, one epoch | Passed; GPU count 0 | 6.04 |

Four input-contract tests passed. The prepared-folder route passed for all five models using a **synthetic prepared fixture**; SHA-256 and file-set comparison confirmed all 24 input files unchanged. `pip check` found no broken requirements.

## Validation scope

The default fixture has 32 invented articles, 8 users, 32 training impressions
and 16 validation impressions, each with one positive and four negatives.
DKN additionally has 16 test impressions. NewsRec's four models evaluate on
validation; DKN reports on the separate synthetic test set. These metrics do
not support comparing model quality or reproducing paper results.

All five notebooks train, evaluate, save weights, restore into the **same model
object**, and compare evaluation metrics before/after restoration. This checks
checkpoint round trips, not new-process deployment. The four NewsRec notebooks
also write ranked predictions. Files use temporary directories and are removed
at notebook completion; executed notebooks and result receipts are retained.

The [recorded results](news-validation-results.json) contain measured notebook
times, source hashes, data labels, GPU counts and restoration checks. Notebook
times exclude dependency installation and are single-run measurements. Native
Windows training, GPU training, real MIND, original paper budgets, cloud use and
new-process reload remain unverified. TensorFlow 1-style graph/session and
deprecation warnings remain in the pinned 2.15 runtime.

This workload also checks Minstar's research-reproducibility instructions:
resource access, synthetic execution and real-data performance must be reported
separately. Only this reusable procedure is fed back into the skill library;
no MIND data or pretrained weights are redistributed there.

Superseded development environments and generated input/output directories were removed after retaining final receipts: 2,276,006,146 file bytes across seven directories. The final environment, final executed notebooks, sources, user data and shared caches were preserved. An independent AI reviewer checked source hashes and evidence labels; no material findings remained. That review did not rerun training. No confirmation Actions were started.
