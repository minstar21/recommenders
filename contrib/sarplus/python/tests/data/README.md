# Offline SAR regression data

These fixtures are synthetic and contain no external research or user data.
Regenerate with `python tests/generate_reference_data.py` from the Python package
directory. The generator imports no SARPlus code: it calculates common-user
counts, set-union Jaccard, frequency-product lift, 30-day half-life affinity,
and matrix-product recommendation scores using standard-library arithmetic.

The 9-user, 14-item fixture exercises thresholds 1 and 3, seen-item removal,
and affinity decay after 60 days. It replaces an unavailable default data host;
Two rare items with one and two users exercise threshold filtering.
it does not reproduce the original external corpus or its coverage. To run
against an independently retained copy of the original CSVs, set
`SARPLUS_TEST_DATA_DIR` to its directory (or a URL ending in the dataset path).
The original assertions remain active for either dataset.
