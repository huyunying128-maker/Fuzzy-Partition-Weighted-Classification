# Changelog

All notable changes to this repository are documented in this file.

The format follows a simple release log with sections for added, changed, fixed, and removed items.

## Unreleased

### Added

- Python package source under `src/fpwc/`.
- Crisp and fuzzy partition utilities.
- Distance-table and fuzzy membership computation.
- Truncation criteria: distance-table difference, Shannon entropy, harmonic distance-change control, square probability, and HPD.
- Feature construction utilities for local weighted inputs and fuzzy-incorporated external classifier inputs.
- Local fuzzy partition-weighted logit classifier.
- External classifier comparison utilities.
- MNIST data preparation and experiment scripts.
- Ablation, truncation, external classifier, centroid visualization, and summary table scripts.
- Pytest test suite for core numerical functions.
- GitHub Actions workflow for automated tests.
- Project documentation directories for data, models, outputs, assets, and paper materials.

### Changed

- Repository documentation updated to match the current project structure.

### Fixed

- Initial package structure standardized for editable installation with `pip install -e .`.

### Removed

- No removed items.
