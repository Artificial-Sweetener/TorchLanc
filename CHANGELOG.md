# 1.1.1 (2025-11-07)

### Fixed

- Respects `color_space="srgb"` across the initial run and OOM probing so cached profiles and output match the requested color space.
- Records the effective auto-selected chunk size in the memory profile cache to avoid reusing `0` and throttling subsequent resizes.

# 1.1.0 (2025-10-12)

### Added

- `color_space` parameter to `lanczos_resize` to allow resampling in either `linear` (default) or `srgb` color space.
- New benchmark options to process all images in a directory as a single batch.

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
