# 1.1.1 (2025-11-07)

### Changed

- `run_benchmark_race.(sh|bat)` now bootstrap virtualenvs with CUDA 13 nightly PyTorch wheels by default (CPU nightly fallback when kernels are unavailable).
- `torchlanc/torchlanc.py` comments were pruned and docstrings expanded so the module is self-documenting without redundant inline narration.
- Added regression tests covering color-space selection, chunk-size auto defaults, alpha isolation, cache behavior, and the TORCHLANC_VALIDATE_RANGE knob.
- README now documents advanced environment controls for VRAM budgeting and range validation.

### Fixed

- Respects `color_space="srgb"` across the initial run and OOM probing so cached profiles and output match the requested color space.
- Records the effective auto-selected chunk size in the memory profile cache to avoid reusing `0` and throttling subsequent resizes.

# 1.1.0 (2025-10-12)

### Added

- `color_space` parameter to `lanczos_resize` to allow resampling in either `linear` (default) or `srgb` color space.
- New benchmark options to process all images in a directory as a single batch.
- Documented advanced environment controls (`TORCHLANC_VRAM_FRACTION`, `TORCHLANC_VALIDATE_RANGE`).

### Changed

- The version was bumped to 1.1.0.
- `python_requires` is now `>=3.8`.

# 1.0.0 (2025-08-22)

### Added

- Initial release of the `torchlanc` library.
- High-quality, separable Lanczos resampler for PyTorch.
- Support for gamma-correct resizing.
- Support for images with alpha channels.
- Persistent caching of Lanczos weights for improved performance.

### Changed

- Restructured the project for publishing to PyPI.
- Updated `readme.md` with installation instructions.

### Fixed

- Corrected issues with the `setup.cfg` file and other project configuration.
