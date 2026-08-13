# Reproducibility

- All train/validation/test splits are chronological.
- Feature scaling is fitted on training data only.
- Seeds are set for Python, NumPy, and PyTorch.
- Experiment parameters live in YAML configuration files.
- Synthetic data are labelled and never merged into real benchmark tables.
- Final research artifacts should be created by scripts, not edited manually.
- `scripts/hash_artifacts.py` generates SHA-256 provenance for report outputs.
- Raw third-party data are not committed to this repository.

## Recommended confirmatory protocol

1. Freeze raw data file hashes.
2. Freeze sensor inclusion criteria.
3. Freeze split boundaries.
4. Run baselines once per registered seed set.
5. Run the proposed model under the same splits and seeds.
6. Run ablations and sensor-failure tests.
7. Export machine-readable results.
8. Perform paired statistical tests across sensors/horizons/seeds.
9. Generate the website findings page from exported results.
