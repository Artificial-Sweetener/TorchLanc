"""
A standalone, pure-PyTorch implementation of a high-quality, separable Lanczos resampler.
This module is designed to be self-contained and run efficiently on a GPU.
Features gamma-correct resizing, a configurable chunk size, and a persistent cache.
"""

import json
import logging
import math
import os
import pathlib
import platform
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import torch

# --- Setup ---
logger = logging.getLogger(__name__)


class LanczosCache:
    """A thread-safe, size-limited LRU cache for Lanczos weight tensors."""

    def __init__(self, max_size_mb: int):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _get_tensor_size_bytes(
        self, tensor_tuple: Tuple[torch.Tensor, torch.Tensor]
    ) -> int:
        """Calculates the total memory usage of the tensors in the cache entry."""
        weights, indices = tensor_tuple
        return (
            weights.element_size() * weights.nelement()
            + indices.element_size() * indices.nelement()
        )

    def get(self, key: tuple) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieves an item from the cache and marks it as recently used."""
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: tuple, value: Tuple[torch.Tensor, torch.Tensor]):
        """Adds an item to the cache, evicting old items if size limit is exceeded."""
        with self.lock:
            if key in self.cache:
                old_value = self.cache[key]
                self.current_size_bytes -= self._get_tensor_size_bytes(old_value)

            value_size = self._get_tensor_size_bytes(value)

            while self.current_size_bytes + value_size > self.max_size_bytes:
                if not self.cache:
                    break
                evicted_key, evicted_value = self.cache.popitem(last=False)
                evicted_size = self._get_tensor_size_bytes(evicted_value)
                self.current_size_bytes -= evicted_size
                logger.debug(
                    "Lanczos cache: Evicted %s (%.2f KB) to free space.",
                    evicted_key,
                    evicted_size / 1024.0,
                )

            self.cache[key] = value
            self.current_size_bytes += value_size
            self.cache.move_to_end(key)

    def state_dict(self) -> dict:
        """Returns the cache's internal dictionary for saving."""
        with self.lock:
            return dict(self.cache)

    def load_state_dict(self, state_dict: dict):
        """Loads the cache from a dictionary."""
        with self.lock:
            self.cache = OrderedDict(state_dict)
            # Recalculate the current size based on the loaded items
            self.current_size_bytes = sum(
                self._get_tensor_size_bytes(v) for v in self.cache.values()
            )
            logger.info(
                f"Lanczos cache: Loaded {len(self.cache)} items, totaling {self.current_size_bytes / (1024*1024):.2f} MB."
            )


# --- Persistent, Thread-safe Weight Caching ---
_lanczos_cache = LanczosCache(max_size_mb=32)
_memory_profile_cache = {}  # In-memory memory profiles
_weights_cache_dirty = False  # set True when new weight entries are added


def _default_cache_root() -> pathlib.Path:
    env = os.environ.get("TORCHLANC_CACHE_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        return pathlib.Path(base) / "torchlanc" / "cache"
    if system == "Darwin":
        return pathlib.Path.home() / "Library" / "Caches" / "torchlanc"
    return pathlib.Path.home() / ".cache" / "torchlanc"


_cache_dir = _default_cache_root()
_lanczos_weights_cache_file = _cache_dir / "lanczos_weights.pt"
_memory_profile_cache_file = _cache_dir / "memory_profile.json"


def _atomic_replace(src: pathlib.Path, dst: pathlib.Path) -> None:
    os.replace(src, dst)


def _save_memory_profile_cache() -> None:
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        tmp = _memory_profile_cache_file.with_suffix(
            _memory_profile_cache_file.suffix + ".tmp"
        )
        payload = {"version": 2, "profiles": _memory_profile_cache}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _atomic_replace(tmp, _memory_profile_cache_file)
    except Exception as e:
        logger.error(f"Memory profile cache: Could not save. Error: {e}")


def _load_memory_profile_cache() -> None:
    """Load memory profiles and migrate schema → v2 if needed."""
    global _memory_profile_cache
    if not _memory_profile_cache_file.exists():
        return
    try:
        with open(_memory_profile_cache_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict) or "profiles" not in payload:
            logger.warning("Memory profile cache: Bad schema. Starting fresh.")
            _memory_profile_cache = {}
            return

        version = int(payload.get("version", 1))
        profiles = payload["profiles"]
        if not isinstance(profiles, dict):
            logger.warning("Memory profile cache: Profiles not a dict. Starting fresh.")
            _memory_profile_cache = {}
            return

        # Migrate v1 → v2 (rename 'optimal_chunk_size' → 'max_safe_chunk')
        if version == 1:
            migrated = {}
            for k, v in profiles.items():
                if isinstance(v, dict):
                    out = dict(v)
                    if "max_safe_chunk" not in out and "optimal_chunk_size" in out:
                        out["max_safe_chunk"] = out.pop("optimal_chunk_size")
                else:
                    out = v
                migrated[k] = out
            _memory_profile_cache = migrated
            logger.info(
                f"Memory profile cache: Migrated {len(_memory_profile_cache)} entries to v2."
            )
            # Persist the migrated data as v2
            _save_memory_profile_cache()
            return

        # v2 already
        _memory_profile_cache = profiles
        logger.info(
            f"Memory profile cache: Loaded {len(_memory_profile_cache)} entries."
        )
    except Exception as e:
        logger.warning(
            f"Memory profile cache: Could not load. Starting fresh. Error: {e}"
        )
        _memory_profile_cache = {}


def _save_cache() -> None:
    """
    Persist weight/indices in CPU tensors with a schema header.
    On disk, keys are device-agnostic to maximize reuse across devices/dtypes.
    """
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        tmp = _lanczos_weights_cache_file.with_suffix(
            _lanczos_weights_cache_file.suffix + ".tmp"
        )
        entries_cpu = OrderedDict()
        with _lanczos_cache.lock:
            for k, (w, idx) in _lanczos_cache.cache.items():
                # Device-specific in-memory key shape:
                # ("lanczos_wts", in_size, out_size, a, str(dtype), device_tag)
                base_key = None
                if isinstance(k, tuple) and len(k) >= 4 and k[0] == "lanczos_wts":
                    base_key = ("lanczos_wts", k[1], k[2], k[3])
                # Fallback: keep original key if it doesn't match expected pattern
                target_key = base_key if base_key is not None else k

                if target_key in entries_cpu:
                    continue  # already captured a CPU copy for this base key

                w_cpu = w.detach().to(device="cpu", dtype=torch.float32)
                idx_cpu = idx.detach().to(device="cpu")
                entries_cpu[target_key] = (w_cpu, idx_cpu)

        payload = {"version": 1, "entries": entries_cpu}
        torch.save(payload, tmp)
        _atomic_replace(tmp, _lanczos_weights_cache_file)
    except Exception as e:
        logger.error(f"Lanczos cache: Could not save. Error: {e}")


def _load_cache() -> None:
    if not _lanczos_weights_cache_file.exists():
        return
    try:
        payload = torch.load(_lanczos_weights_cache_file, map_location="cpu")
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or "entries" not in payload
        ):
            logger.warning("Lanczos cache: Bad schema. Starting fresh.")
            return
        entries = payload["entries"]
        if not isinstance(entries, dict):
            logger.warning("Lanczos cache: Entries not a dict. Starting fresh.")
            return
        # Load entries as-is (CPU tensors). In-memory LRU will move them to the target device on demand.
        _lanczos_cache.load_state_dict(entries)
        logger.info(f"Lanczos cache: Loaded {len(entries)} items.")
    except Exception as e:
        logger.warning(f"Lanczos cache: Could not load. Starting fresh. Error: {e}")


_load_cache()
_load_memory_profile_cache()


def _get_device_identifier() -> str:
    """Returns a stable identifier for the active device (CUDA UUID or a CPU signature)."""
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
        except Exception:
            idx = 0
        try:
            uuid = torch.cuda.get_device_properties(idx).uuid
            return f"cuda_{uuid}"
        except Exception:
            return f"cuda_device_{idx}"
    try:
        u = platform.uname()
        ident = "_".join(filter(None, [u.system, u.machine, u.processor])).replace(
            " ", "_"
        )
        return f"cpu_{ident}" if ident else "cpu_generic"
    except Exception:
        return "cpu_generic"


def _suggest_chunk_rows(
    b: int,
    c: int,
    in_h: int,
    in_w: int,
    out_h: int,
    out_w: int,
    a: int,
    device: torch.device,
) -> int:
    """
    Suggest a conservative row-chunk for _resample_1d_jit based on current free VRAM.
    Rows = B * C * other_dim for each pass; chunk_size is in rows.
    """
    if device.type != "cuda":
        return 2**31 - 1  # effectively "no constraint" on CPU

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return 2**20  # reasonable default if query fails

    # Budget: fraction of currently free memory; tunable via env.
    frac = float(os.environ.get("TORCHLANC_VRAM_FRACTION", "0.30"))
    frac = max(0.05, min(frac, 0.9))
    budget = max(64 * 1024 * 1024, int(free_bytes * frac))

    # Lanczos support window (largest across width/height passes).
    scale_w = max(float(in_w) / float(out_w), 1.0)
    scale_h = max(float(in_h) / float(out_h), 1.0)
    win_w = int(math.ceil(a * scale_w) * 2)
    win_h = int(math.ceil(a * scale_h) * 2)
    window = max(win_w, win_h)

    # Element size: resample runs in the tensor's compute dtype; default is fp32.
    elem_size = 4  # float32 bytes; if you enable fp16/bf16, adjust here accordingly.

    # Ephemeral per chunk: gathered_pixels ~ rows * window * elem_size
    # Keep at most half the budget for this term to leave room for other temporaries.
    if window <= 0:
        return 2**31 - 1
    rows_budget = max(1, (budget // 2) // max(1, window * elem_size))

    # Also can’t exceed true row counts; take the min across both passes.
    rows_width_pass = b * c * in_h
    rows_height_pass = b * c * out_w
    hard_cap = min(rows_width_pass, rows_height_pass)

    return max(1, min(rows_budget, hard_cap))


# --- Gamma Correction Functions ---
def srgb_to_linear(tensor: torch.Tensor) -> torch.Tensor:
    """Converts a tensor from sRGB to linear color space."""
    return torch.where(
        tensor <= 0.04045, tensor / 12.92, ((tensor + 0.055) / 1.055).pow(2.4)
    )


def linear_to_srgb(tensor: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Converts a tensor from linear to sRGB color space."""
    srgb = torch.where(
        tensor <= 0.0031308, tensor * 12.92, 1.055 * (tensor.pow(1.0 / 2.4)) - 0.055
    )
    if clamp:
        return srgb.clamp(0.0, 1.0)
    return srgb


@torch.jit.script
def lanczos_kernel_1d(x: torch.Tensor, a: int = 3) -> torch.Tensor:
    """Computes the 1D Lanczos kernel L(x) for a tensor of distances `x`."""
    x = x.to(torch.float32)
    is_in_domain = torch.abs(x) < a
    kernel = torch.sinc(x) * torch.sinc(x / a)
    return torch.where(
        is_in_domain, kernel, torch.tensor(0.0, device=x.device, dtype=x.dtype)
    )


@torch.jit.script
def _resample_1d_jit(
    tensor: torch.Tensor,
    in_size: int,
    out_size: int,
    dim: int,
    chunk_size: int,
    weights: torch.Tensor,
    clamped_indices: torch.Tensor,
) -> torch.Tensor:
    tensor_reshaped = tensor.movedim(dim, -1)
    original_shape = tensor_reshaped.shape
    tensor_flat = tensor_reshaped.reshape(-1, in_size)

    num_rows = tensor_flat.shape[0]
    if chunk_size >= num_rows:
        gathered_pixels = tensor_flat[:, clamped_indices]
        resampled_flat = (gathered_pixels * weights.unsqueeze(0)).sum(dim=-1)
    else:
        resampled_flat = torch.empty(
            (num_rows, out_size), device=tensor_flat.device, dtype=tensor_flat.dtype
        )
        num_chunks = (num_rows + chunk_size - 1) // chunk_size
        w0 = weights.unsqueeze(0)
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_rows)
            chunk = tensor_flat[start:end]
            gathered_pixels = chunk[:, clamped_indices]
            resampled_flat[start:end] = (gathered_pixels * w0).sum(dim=-1)

    new_shape = list(original_shape)
    new_shape[-1] = out_size
    return resampled_flat.reshape(new_shape).movedim(-1, dim)


def resample_1d(
    tensor: torch.Tensor, out_size: int, a: int, dim: int, chunk_size: int
) -> torch.Tensor:
    """Resamples `tensor` along `dim` using a separable Lanczos kernel."""
    global _weights_cache_dirty

    compute_dtype = tensor.dtype
    in_size = tensor.shape[dim]
    device = tensor.device

    # Build per-device cache key; base key is device-agnostic for disk persistence.
    if tensor.is_cuda:
        idx = tensor.device.index if tensor.device.index is not None else 0
        try:
            dev_uuid = torch.cuda.get_device_properties(idx).uuid
            device_tag = f"cuda:{dev_uuid}"
        except Exception:
            device_tag = f"cuda:{idx}"
    else:
        device_tag = "cpu"

    device_key = ("lanczos_wts", in_size, out_size, a, str(compute_dtype), device_tag)
    base_key = ("lanczos_wts", in_size, out_size, a)

    # Resolve weights/indices locally and ensure correct device/dtype.
    weights: torch.Tensor
    clamped_indices: torch.Tensor

    cached = _lanczos_cache.get(device_key)
    if cached is not None:
        w, idx = cached
        weights = w.to(device=device, dtype=compute_dtype, copy=False)
        clamped_indices = idx.to(device=device, copy=False)
    else:
        base_cached = _lanczos_cache.get(base_key)
        if base_cached is not None:
            w_cpu, idx_cpu = base_cached
            weights = w_cpu.to(device=device, dtype=compute_dtype, copy=False)
            clamped_indices = idx_cpu.to(device=device, copy=False)
            _lanczos_cache.put(device_key, (weights, clamped_indices))
            # Remove base to avoid duplication in RAM.
            try:
                with _lanczos_cache.lock:
                    if base_key in _lanczos_cache.cache:
                        _lanczos_cache.current_size_bytes -= (
                            _lanczos_cache._get_tensor_size_bytes(
                                _lanczos_cache.cache[base_key]
                            )
                        )
                        del _lanczos_cache.cache[base_key]
            except Exception:
                pass
        else:
            # Build weights on the target device.
            scale = float(in_size) / float(out_size)
            kernel_scale = max(scale, 1.0)
            support = float(a) * kernel_scale
            window_size = int(torch.ceil(torch.tensor(support) * 2).item())

            out_coords = (
                torch.arange(out_size, device=device, dtype=torch.float32) + 0.5
            ) * scale - 0.5
            start_indices = (out_coords - support + 0.5).floor().to(torch.int64)
            window_indices = start_indices.unsqueeze(1) + torch.arange(
                window_size, device=device, dtype=torch.int64
            ).unsqueeze(0)

            distances = out_coords.unsqueeze(1) - window_indices.to(torch.float32)
            scaled_distances = distances / kernel_scale
            weights_f32 = lanczos_kernel_1d(scaled_distances, a=a)
            weights_sum = weights_f32.sum(dim=1, keepdim=True)
            weights_f32 = torch.where(
                weights_sum == 0, weights_f32, weights_f32 / weights_sum
            )

            weights = weights_f32.to(compute_dtype)
            clamped_indices = torch.clamp(window_indices, 0, in_size - 1)

            _lanczos_cache.put(device_key, (weights, clamped_indices))
            _weights_cache_dirty = True

    # Use the local, device-correct tensors; do not re-fetch from the LRU here.
    return _resample_1d_jit(
        tensor, in_size, out_size, dim, chunk_size, weights, clamped_indices
    )


def _lanczos_resize_core(
    image_tensor: torch.Tensor,
    height: int,
    width: int,
    a: int = 3,
    chunk_size: int = 2048,
    clamp: bool = True,
    precision: str = "high",
    color_space: str = "linear",
) -> torch.Tensor:
    """
    Resizes a batch of images using a separable, gamma-correct Lanczos filter.
    No internal retries or memory cleanup; any OOM propagates to the caller.
    """
    if not (image_tensor.ndim == 4 and image_tensor.shape[1] in [1, 3, 4]):
        raise ValueError(
            "Input must be a 4D tensor (B, C, H, W) with 1, 3, or 4 channels."
        )

    if chunk_size <= 0:
        chunk_size = 65536 if image_tensor.device.type == "cpu" else 2048

    original_dtype = image_tensor.dtype

    # Precision selection for resampling; gamma is always processed in float32
    _precision_map = {
        "high": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
        "medium": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    compute_dtype = _precision_map.get(precision.lower(), torch.float32)

    if color_space not in ("linear", "srgb"):
        raise ValueError("color_space must be 'linear' or 'srgb'")

    if image_tensor.shape[1] == 4:
        rgb, alpha = torch.split(image_tensor, [3, 1], dim=1)
    else:
        rgb, alpha = image_tensor, None

    if color_space == "linear":
        # sRGB -> linear in float32 for accuracy
        linear_rgb_f32 = srgb_to_linear(rgb.to(torch.float32))
        linear_rgb = linear_rgb_f32.to(compute_dtype)

        # Separable resampling: width then height (in linear)
        resized_w = resample_1d(linear_rgb, width, a=a, dim=-1, chunk_size=chunk_size)
        resized_linear_rgb = resample_1d(
            resized_w, height, a=a, dim=-2, chunk_size=chunk_size
        )

        # linear -> sRGB in float32; stream slice-by-slice to avoid large temporaries
        b, c, h, w = resized_linear_rgb.shape
        bytes_per_img_f32 = c * h * w * 4
        budget = (
            256 * 1024 * 1024 if image_tensor.device.type == "cuda" else 1024 * 1024 * 1024
        )
        gamma_chunk_size = max(1, min(b, int(budget // max(1, bytes_per_img_f32))))

        if resized_linear_rgb.dtype == torch.float32:
            if gamma_chunk_size >= b:
                resized_linear_rgb.copy_(linear_to_srgb(resized_linear_rgb, clamp=clamp))
            else:
                num_chunks = (b + gamma_chunk_size - 1) // gamma_chunk_size
                for i in range(num_chunks):
                    s = i * gamma_chunk_size
                    e = min((i + 1) * gamma_chunk_size, b)
                    out_slice = linear_to_srgb(resized_linear_rgb[s:e], clamp=clamp)
                    resized_linear_rgb[s:e].copy_(out_slice)
            resized_srgb = resized_linear_rgb
        else:
            if gamma_chunk_size >= b:
                out_slice = linear_to_srgb(resized_linear_rgb.to(torch.float32), clamp=clamp)
                resized_linear_rgb.copy_(out_slice.to(resized_linear_rgb.dtype))
            else:
                num_chunks = (b + gamma_chunk_size - 1) // gamma_chunk_size
                for i in range(num_chunks):
                    s = i * gamma_chunk_size
                    e = min((i + 1) * gamma_chunk_size, b)
                    out_slice = linear_to_srgb(resized_linear_rgb[s:e].to(torch.float32), clamp=clamp)
                    resized_linear_rgb[s:e].copy_(out_slice.to(resized_linear_rgb.dtype))
            resized_srgb = resized_linear_rgb
    else:
        # Resample directly in sRGB (gamma-encoded) space to match FFmpeg/Pillow
        rgb_compute = rgb.to(compute_dtype)
        resized_w = resample_1d(rgb_compute, width, a=a, dim=-1, chunk_size=chunk_size)
        resized_srgb = resample_1d(
            resized_w, height, a=a, dim=-2, chunk_size=chunk_size
        )


    if alpha is not None:
        alpha_compute = alpha.to(compute_dtype)
        resized_alpha_w = resample_1d(
            alpha_compute, width, a=a, dim=-1, chunk_size=chunk_size
        )
        resized_alpha = resample_1d(
            resized_alpha_w, height, a=a, dim=-2, chunk_size=chunk_size
        )
        resized_image = torch.cat(
            (resized_srgb, resized_alpha.to(resized_srgb.dtype)), dim=1
        )
    else:
        resized_image = resized_srgb

    return resized_image.to(original_dtype)


def lanczos_resize(
    image_tensor: torch.Tensor,
    height: int,
    width: int,
    a: int = 3,
    chunk_size: int = 2048,
    clamp: bool = True,
    precision: str = "high",
    color_space: str = "linear",
) -> torch.Tensor:

    """
    Resizes a batch of images using a separable, gamma-correct Lanczos filter.
    Handles OOM by adaptively splitting batches and backing off chunk size.
    """
    global _weights_cache_dirty

    if not (image_tensor.ndim == 4 and image_tensor.shape[1] in [1, 3, 4]):
        raise ValueError(
            "Input must be a 4D tensor of shape (B, C, H, W) with 1, 3, or 4 channels."
        )
    if not image_tensor.is_floating_point():
        raise ValueError("Input tensor must be floating point in [0, 1].")

    # Optional range validation to avoid heavy GPU sync on hot paths.
    if os.environ.get("TORCHLANC_VALIDATE_RANGE") == "1":
        if torch.any(image_tensor < 0) or torch.any(image_tensor > 1):
            raise ValueError("Input tensor values must be in [0, 1].")

    initial_batch_size = image_tensor.shape[0]
    _, _, input_height, input_width = image_tensor.shape

    # --- Generate Cache Key ---
    device_id = _get_device_identifier()
    cache_key = f"{device_id}::{input_height}::{input_width}::{height}::{width}::{a}::{precision}::{color_space}"


    optimal_batch_size = None
    optimal_chunk_size = None

    # --- Try to load from cache ---
    if cache_key in _memory_profile_cache:
        cached_profile = _memory_profile_cache[cache_key]
        optimal_batch_size = int(cached_profile.get("optimal_batch_size", 0))
        # v2: max_safe_chunk; v1 fallback: optimal_chunk_size
        max_safe_chunk = int(
            cached_profile.get(
                "max_safe_chunk", cached_profile.get("optimal_chunk_size", 0)
            )
        )
        logger.info(
            f"Memory profile cache hit for {cache_key}: "
            f"optimal_batch_size={optimal_batch_size}, max_safe_chunk={max_safe_chunk}"
        )

        if optimal_batch_size > 0 and max_safe_chunk > 0:
            suggested = _suggest_chunk_rows(
                initial_batch_size,
                image_tensor.shape[1],
                image_tensor.shape[2],
                image_tensor.shape[3],
                height,
                width,
                a,
                image_tensor.device,
            )
            effective_chunk = max(1, min(max_safe_chunk, suggested))

            if optimal_batch_size >= initial_batch_size:
                out = _lanczos_resize_core(
                    image_tensor, height, width, a, effective_chunk, clamp, precision, color_space
                )
                if _weights_cache_dirty:
                    _save_cache()
                    _weights_cache_dirty = False
                return out

            out = torch.empty(
                (initial_batch_size, image_tensor.shape[1], height, width),
                dtype=image_tensor.dtype,
                device=image_tensor.device,
            )
            for i in range(0, initial_batch_size, optimal_batch_size):
                sub_batch = image_tensor[i : i + optimal_batch_size]
                processed = _lanczos_resize_core(
                    sub_batch, height, width, a, effective_chunk, clamp, precision, color_space
                )
                out[i : i + processed.shape[0]] = processed

            if _weights_cache_dirty:
                _save_cache()
                _weights_cache_dirty = False
            return out

    # --- Adaptive execution: initial full-batch attempt, then layered search on OOM ---
    try:
        resized_image = _lanczos_resize_core(
            image_tensor, height, width, a, chunk_size, clamp, precision
        )
        _memory_profile_cache[cache_key] = {
            "optimal_batch_size": initial_batch_size,
            "max_safe_chunk": chunk_size,
        }
        _save_memory_profile_cache()
        if _weights_cache_dirty:
            _save_cache()
            _weights_cache_dirty = False
        return resized_image

    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        found_batch = 0
        found_chunk = 0
        start_batch = max(1, initial_batch_size // 2)
        max_chunk_retries = 20

        current_batch = start_batch
        while current_batch >= 1 and found_batch == 0:

            # Phase A: start from min(cached upper bound, current suggestion)
            cached_profile = _memory_profile_cache.get(cache_key, {})
            cached_bound = int(
                cached_profile.get(
                    "max_safe_chunk",
                    cached_profile.get("optimal_chunk_size", chunk_size),
                )
            )
            suggested = _suggest_chunk_rows(
                current_batch,
                image_tensor.shape[1],
                image_tensor.shape[2],
                image_tensor.shape[3],
                height,
                width,
                a,
                image_tensor.device,
            )
            test_chunk = max(1, int(min(cached_bound, suggested)))
            ok = False

            for _ in range(max_chunk_retries):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    _ = _lanczos_resize_core(
                        image_tensor[:current_batch],
                        height,
                        width,
                        a,
                        test_chunk,
                        clamp,
                        precision,
                    )
                    ok = True
                    break
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    if test_chunk == 1:
                        ok = False
                        break
                    test_chunk = max(1, test_chunk // 2)

            if not ok:
                current_batch //= 2
                continue

            # Phase B: grow by ×2 to the largest safe chunk for speed
            grown = test_chunk
            for _ in range(max_chunk_retries):
                probe = grown * 2
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    _ = _lanczos_resize_core(
                        image_tensor[:current_batch],
                        height,
                        width,
                        a,
                        probe,
                        clamp,
                        precision,
                    )
                    grown = probe
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                    break

            found_batch = current_batch
            found_chunk = grown
            # done with this batch size
            break

        if found_batch == 0:
            _memory_profile_cache[cache_key] = {
                "optimal_batch_size": 0,
                "max_safe_chunk": 0,
            }
            _save_memory_profile_cache()
            raise torch.OutOfMemoryError(
                "Could not find a safe batch size for image processing."
            )

        out = torch.empty(
            (initial_batch_size, image_tensor.shape[1], height, width),
            dtype=image_tensor.dtype,
            device=image_tensor.device,
        )
        for i in range(0, initial_batch_size, found_batch):
            sub_batch = image_tensor[i : i + found_batch]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            processed = _lanczos_resize_core(
                sub_batch, height, width, a, found_chunk, clamp, precision, color_space
            )
            out[i : i + processed.shape[0]] = processed

        _memory_profile_cache[cache_key] = {
            "optimal_batch_size": found_batch,
            "max_safe_chunk": found_chunk,
        }
        _save_memory_profile_cache()
        if _weights_cache_dirty:
            _save_cache()
            _weights_cache_dirty = False
        return out

    except Exception as e:
        logger.error(f"An unexpected error occurred during lanczos_resize: {e}")
        raise  # Re-raise any other exceptions


# --- Public utilities & facade ---


def clear_weight_cache(persist: bool = True) -> None:
    with _lanczos_cache.lock:
        _lanczos_cache.cache.clear()
        _lanczos_cache.current_size_bytes = 0
    if persist:
        try:
            os.remove(_lanczos_weights_cache_file)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"clear_weight_cache: failed to remove file: {e}")


def clear_profile_cache(persist: bool = True) -> None:
    global _memory_profile_cache
    _memory_profile_cache = {}
    if persist:
        try:
            os.remove(_memory_profile_cache_file)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"clear_profile_cache: failed to remove file: {e}")


def set_cache_dir(path: str, reload_from_disk: bool = True) -> pathlib.Path:
    global _cache_dir, _lanczos_weights_cache_file, _memory_profile_cache_file, _memory_profile_cache
    new_dir = pathlib.Path(path).expanduser().resolve()
    os.makedirs(new_dir, exist_ok=True)
    _cache_dir = new_dir
    _lanczos_weights_cache_file = _cache_dir / "lanczos_weights.pt"
    _memory_profile_cache_file = _cache_dir / "memory_profile.json"

    if reload_from_disk:
        clear_weight_cache(persist=False)
        clear_profile_cache(persist=False)
        _load_cache()
        _load_memory_profile_cache()

    return _cache_dir


# Primary facade + alias
resize = lanczos_resize

__all__ = [
    "lanczos_resize",
    "resize",
    "clear_weight_cache",
    "clear_profile_cache",
    "set_cache_dir",
]
