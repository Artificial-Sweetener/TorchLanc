# **TorchLanc: A High-Quality Lanczos Scaler for PyTorch**

I built this because I needed a resampler that was **fast, principled, and uncompromising in quality**.
TorchLanc is for anyone who cares about the details:
true anti-aliasing, gamma-correct math, and GPU-ready speed; all in a single, self-contained module you can drop straight into your project.

---

## **Features**

### **Resizing Done Right: Sinc and Lanczos**

Every image is a **grid of samples** - some captured from the real world, some painted pixel by pixel, or even dreamed up by a neural net.
Whatever the source, resizing is about building a new grid that’s faithful to the original one - you want to change the size without losing fidelity.

The **sinc function** is the theoretical ideal for accomplishing tat. Imagine every pixel in your image as a point on an endless graph, and sinc as the perfect way to connect those points into a smooth, consistent signal; everywhere, forever.  But sinc is infinite. You see, each pixel whispers to every other pixel, no matter how far away; every point on the graph has some impact on every other point. To calculate even one new pixel, you’d have to listen to all of them, across the entire image, into infinity.

That’s where **Lanczos** comes in.
Lanczos draws a **local circle** called the **sinc window** around each pixel, wide enough to capture the important whispers nearby, while ignoring the ones too faint to make a perceptible difference.
The result is sharp, natural resizing; a practical balance between perfection and performance.

---

### **Gamma-Correct Color**

Most images are stored in **sRGB**, where the numbers in the file don’t match the true intensity of light. Doing math directly on those numbers is like trying to mix paint in a room lit by colored bulbs... the math is wrong before you even start!

TorchLanc converts your pixels into **linear light** before resampling, then brings them back to sRGB afterward.
It’s like stepping into clean daylight to do your work, then returning to the gallery with your colors intact: bright, true, and vibrant.

---

### **GPU Acceleration**

A CPU is like a **skilled soloist**; precise, flexible, able to improvise through complex logic one note at a time.
A GPU is like a **massive orchestra**, hundreds of players performing the same score in perfect synchrony.

For resizing, that score is simple and repetitive: multiply, sum, repeat. The GPU thrives on that kind of harmony, processing thousands of pixels in parallel while the CPU would still be playing the opening bars.

TorchLanc leans into that parallelism, letting the GPU carry the heavy, uniform work while keeping results sharp and consistent.

---

### **Persistent Weight Cache**

When TorchLanc resizes between two specific sizes for the first time, it solves the heavy math problem of calculating the ideal resampling weights; that's the “recipe” for that exact transformation.

Then it writes that recipe down in a cache.
Next time you need the same resize, you don't have to do the calculation over again! Thanks to the cache, you can have instant, precise reuse of the math you already did. It’s **memoization for image processing**, baked in.

---

## **Usage**

```python
import torch
from torchlanc import lanczos_resize

# Batch of images: (B, C, H, W)
my_image_batch = torch.randn(4, 3, 256, 256).to("cuda")

# High-quality resize
resized = lanczos_resize(my_image_batch, height=512, width=512)

# Sharper variant with a=2
resized_sharp = lanczos_resize(my_image_batch, height=512, width=512, a=2)
```

---

## **Parameters**

| Argument        | Type           | Description                                                                 |
|-----------------|---------------|----------------------------------------------------------------------------|
| `image_tensor`  | `torch.Tensor` | A 4D tensor `(B, C, H, W)` of float data in `[0, 1]`.                      |
| `height`        | `int`         | Target height.                                                              |
| `width`         | `int`         | Target width.                                                               |
| `a`             | `int` (opt)   | Lanczos kernel window size. Default `3` is balanced; `2` is sharper; `4` softer. |
| `chunk_size`    | `int` (opt)   | Controls memory chunking. Default `2048` is safe for most jobs. Set `-1` to auto-tune for maximum GPU throughput (~90% of free VRAM). |

## **Installation**

You can install `torchlanc` using pip:

### From PyPI (Recommended)

```bash
pip install torchlanc
```

### From Source (for development)

If you want to install the latest development version or contribute to the project, you can install it directly from GitHub:

```bash
git clone https://github.com/Artificial-Sweetener/TorchLanc.git
cd TorchLanc
pip install .
```

To install with development dependencies (for running tests and benchmarks):

```bash
pip install ".[dev]"
```

---

# Benchmarking

TorchLanc comes with a simple test harness so you can see how it performs. You can compare it against Pillow’s CPU-based Lanczos or just measure TorchLanc on its own.

## What you need

TorchLanc itself does not depend on Pillow. The benchmark script does, along with a couple of helpers.

To run the benchmark, install:
- Pillow (for the CPU comparison)
- torchvision (to load and save images)
- (Optional) py-cpuinfo (to print prettier CPU info)

```bash
pip install pillow torchvision py-cpuinfo
```

## Setting up test images

Put two images in the same folder as `benchmark.py`:
- `test.png` for downscaling tests
- `test2.png` for upscaling tests

You can use any images you like.

## TorchLanc vs. Pillow

To see how TorchLanc compares to Pillow, run:

```bash
python benchmark.py --race
```

This will:
- Resize both up and down for multiple batch sizes
- Time TorchLanc cold (first run, no cache)
- Time TorchLanc warm (cached weights)
- Time Pillow’s CPU Lanczos for comparison
- Save PNG files so you can visually inspect the results

## Self-benchmark

To measure TorchLanc without Pillow:

```bash
# Full series of batch sizes
python benchmark.py --self

# Specific batch size
python benchmark.py --self --batch 256

# Specific operation
python benchmarkt.py --self --op upscale
```

Shortcuts work too:
```bash
python benchmark.py --self-256
python benchmark.py --self-256-upscale
```

## Useful flags

| Flag | Description |
|-------|-------------|
| `--race` | Runs TorchLanc vs. Pillow comparison |
| `--self` | Runs TorchLanc-only benchmarks |
| `--batch <int>` | Sets batch size for `--self` |
| `--op {downscale,upscale}` | Picks the operation for `--self` |
| `--cache-dir <path>` | Uses a custom cache directory |
| `--cpu-only` | Forces CPU-only mode |

## What you get

- Timing stats in the console
  - TorchLanc cold (first run)
  - TorchLanc warm (cached)
  - Pillow CPU total time and time per image, plus percentage relative to TorchLanc warm
- Comparison images in the working directory
  - `comparison_batch_{BATCH}_{OP}.png` shows a split panel for each method
  - `comparison_visual_{OP}.png` shows a three-way comparison: Original, TorchLanc, and Pillow

This is not meant to be a synthetic benchmark. It is a practical way to see how TorchLanc performs on your hardware with your images.

---

## **Why TorchLanc**

If you care about:
- Faithful anti-aliased resizing
- True-to-light color processing
- GPU-scale speed without shortcuts
- Smart caching for repeated jobs

…then TorchLanc was built for you.

Fast. Principled. Beautiful.

 - ArtificialSweetener
