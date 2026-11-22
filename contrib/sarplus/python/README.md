# SARplus

Simple Algorithm for Recommendation (SAR) is a neighborhood based
algorithm for personalized recommendations based on user transaction
history. SAR recommends items that are most **similar** to the ones
that the user already has an existing **affinity** for. Two items are
**similar** if the users that interacted with one item are also likely
to have interacted with the other. A user has an **affinity** to an
item if they have interacted with it in the past.

SARplus uses PySpark 3.x and a matching Scala 2.12 SARPlus JAR. Install the
Python package on Linux, then supply the JAR for your Spark version when
starting the Spark session. The wheel contains the native prediction module;
Spark loads the data source from the JAR.
More details can be found at
[sarplus@microsoft/recommenders](https://github.com/microsoft/recommenders/tree/main/contrib/sarplus).

## Build and test locally

Use Linux (or Ubuntu under WSL), Python 3.8–3.11, a C++ compiler,
Python development headers, JDK 11, and sbt. For WSL builds, copy the
source into the Linux filesystem; sbt sockets do not work on Windows mounts.

From the repository's `contrib/sarplus/python` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy<2" "pandas<3" "pyspark==3.5.3" pyarrow pytest scikit-learn build twine
python -m build
python -m twine check dist/*
python -m pip install dist/*.whl
cd ../scala
SPARK_VERSION=3.5.3 HADOOP_VERSION=3.3.4 sbt '++2.12.18!' test package
cd ../python
PYSPARK_PYTHON="$(command -v python)" SPARK_LOCAL_IP=127.0.0.1 python -m pytest tests -q
```

On Ubuntu, source builds need `g++`, `python3-dev`, and `python3-venv`.
The build prepares `pysarplus/VERSION` from `../VERSION`; a source distribution
already contains this file. Keep the Python package and Scala JAR at the same
SARPlus version, and match the JAR's Scala version with Spark.

The regression suite reads bundled synthetic CSVs without network access.
See [fixture formulas and provenance](tests/data/README.md). External original
fixtures remain available through `SARPLUS_TEST_DATA_DIR`. A JAR whose name
ends in `-unsigned.jar` is a CI artifact without a release signature.
