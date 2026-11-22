# First-install validation: CPU and local Spark

Validated on 2026-11-22 in a separate Ubuntu 22.04 WSL virtual environment:
Python 3.10.12, OpenJDK 11, NumPy 1.26.4, pandas 2.3.3, SciPy 1.13.1,
Spark 3.5.9, Cornac 2.2.2, Surprise 1.1.5, scikit-learn 1.7.2,
LightGBM 4.7.0 and category-encoders 2.8.1.

## Installation and execution

Use the root [README first-run commands](../../README.md#verified-first-run-in-this-fork).
On Ubuntu, install Python 3.10, `python3-venv` and JDK 11. A compiler and
Python development headers are needed if pip must build a native dependency
instead of installing a compatible wheel. Use the Linux filesystem for WSL.

`constraints-verified-py310.txt` records the tested numerical, model and notebook
packages. It is a constraint file (`pip -c`), not a full dependency lockfile.
`pip check` passed after installing the final repository with these constraints.

The runner uses the active environment's Python for both notebooks and Spark
workers. It runs every code cell with errors disallowed, checks recorded metrics
for nonfinite values, saves executed notebook copies and JSON results, and
returns a failing exit code when execution fails. It closes kernel processes
and client channels after each notebook. An existing output directory is rejected
so a later run cannot overwrite earlier evidence.

## Results

| Notebook | Data and checks | Result | Seconds |
| --- | --- | --- | ---: |
| SAR | GroupLens MovieLens 100K; split, fit, top-k and evaluation | Passed | 9.19 |
| ALS | MovieLens 100K; fit, unseen-item ranking, rating evaluation; assert no training pairs remain | Passed | 31.38 |
| LightGBM | 2,000 seeded synthetic click rows; both encoding paths, metrics, saved-model prediction equality | Passed | 45.02 |
| Offline CPU models | SVD 60 rating predictions; BPR 900 unseen scores; TF-IDF 12 content recommendations with a known nearest neighbor | Passed | 6.97 |

Times include kernel startup, execution and shutdown, and exclude installation.
MovieLens caches may be warm; these are execution observations, not model speed
comparisons. [Machine-readable results](validation-results.json) include finite
metrics and LF-normalized source hashes. Synthetic results validate execution;
they do not measure Criteo performance or recommendation usefulness for users.

The existing evaluation, Surprise and Cornac unit suites passed **36 tests**
(`tests/unit/recommenders/evaluation/test_python_evaluation.py`,
`tests/unit/recommenders/models/test_surprise_utils.py`,
`tests/unit/recommenders/models/test_cornac_utils.py`).

## Repaired failures

- Spark extra previously admitted Spark 4.0 despite the supported Scala 2.12
  path. It now selects Spark 3.x.
- Cornac 2.3.0 failed import with `GLIBCXX_3.4.32 not found` on Ubuntu 22.04.
  The supported dependency range selects Cornac below 2.3; 2.2.2 passed.
- ALS requested 16GB and all local cores and disabled ambiguous-self-join checks.
  It now defaults to 4GB/two threads and removes seen pairs with a named-key
  anti join. Override `SPARK_MASTER` and `SPARK_MEMORY` explicitly when needed.
- Criteo's historical sample URL redirects to a blog. LightGBM now starts with
  a labeled synthetic dataset; set `DATA_SOURCE = "local"` and
  `CRITEO_SAMPLE_PATH` to a legally obtained tab-separated file for real data.
  The `lgb.train` call uses categorical metadata on the Dataset, compatible with
  LightGBM 4. Splits use disjoint integer positions and copies.
- An independent reviewer found notebook client channels remaining after kernel
  shutdown. The runner now uses nbclient cleanup and stops remaining channels.

## Supported scope

This run validates these six models in this Python 3.10 Linux environment.
The original TF-IDF CORD-19 notebook needs external data and SciBERT; the new
offline text notebook uses `tokenization_method="none"`. No cloud credentials,
GPU or paid API are used. NCF, RBM and FastAI now have separate
[validated CPU profiles](NEURAL_CPU_VALIDATION.md). Other TensorFlow/PyTorch notebooks,
Databricks/Synapse, Windows-native Spark and other Python versions have not
been executed in this run. MovieLens download availability remains external.
No confirmation GitHub Actions were started.
