# Data availability and licensing

This repository does **not** redistribute raw third-party datasets. It includes
only source code, configurations, aggregate/derived result artifacts, and
figures. Users must obtain each dataset from its authoritative source and
comply with the source license and terms.

The manuscript benchmark uses Adult, Diabetes, Ionosphere, Spambase, Phoneme,
QSAR Biodegradation, Sonar, German Credit, Heart Disease, and MAGIC Gamma
Telescope. `hqkm/local_datasets.py` documents accepted filenames, schemas,
labels, and integrity hashes used during the reported runs.

Bundled scikit-learn toy datasets used by smoke/LODO tests are loaded through
scikit-learn. Optional OpenML datasets are downloaded through the public
scikit-learn/OpenML interface and cached outside the repository.

Do not commit downloaded data, participant-level data, credentials, cache
directories, or machine-specific paths. Derived JSON records in `results/`
have been sanitized to retain only dataset filenames, not local directories.
