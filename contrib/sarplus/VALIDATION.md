# Local release validation

The maintained build path is Linux x86-64. On 2026-11-22 this fork was built
with Python 3.10.12, OpenJDK 11, sbt 1.6.2, Scala 2.12.18, Spark 3.5.3,
NumPy 1.26.4, pandas 2.3.3, and PyArrow 25.0.1.

## Reproduction

Follow [build and test locally](python/README.md#build-and-test-locally).
Ubuntu needs `g++`, `python3-dev`, `python3-venv`, and JDK 11.
Build from the Linux filesystem when using WSL. If the default Maven endpoint
is unavailable, configure sbt's repositories to use Maven Central at
`https://repo.maven.apache.org/maven2/`; do not disable artifact checksum checks.

The Python source archive and native wheel passed `python -m build` and
`twine check`. The wheel was installed into a second empty virtual environment,
with `pip check` reporting no dependency conflicts. Scala tests passed (2/2),
and binary JAR, sources JAR, Javadoc JAR, and POM were generated locally.
The Python suite passed 23/23 tests in the clean environment. It exercises native prediction, Spark IO, end-to-end scoring,
similarity thresholds, time decay, ranking, and top-k truncation using bundled
synthetic reference data. See [fixture provenance](python/tests/data/README.md).

The workflow signing step was exercised on the real Scala outputs with a
throwaway test key: all four detached signatures verified. The test key was
destroyed after the check; it is not a release identity. Missing signing keys
produce explicitly named unsigned CI artifacts. Invalid configured keys fail.

API documentation was rebuilt using `jupyter-book build docs --all`: zero
Sphinx warnings. Duplicate imported API members and ambiguous HParams links
were corrected rather than silencing diagnostics.

## Scope

These results cover local Linux execution of SARPlus. They do not certify the
entire upstream notebook library, every Python/Spark matrix combination,
Databricks/Synapse clusters, or production recommendation quality. A public
PyPI/Maven release requires its maintainer's publishing credentials and release
signature. This fork does not publish to those registries. GitHub Pages is
disabled in this fork; generated HTML can be opened locally.

Routine tests run locally. Version tags, scheduled research workflows and
explicit package/deployment workflows keep their existing triggers. No manual
GitHub Actions were started for this validation.
