import argparse
import os
import platform
import re
import shutil
import sys
import time

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

# Attempt to import cpuinfo for detailed CPU specs
try:
    import cpuinfo
except ImportError:
    cpuinfo = None

from torchlanc import (
    clear_profile_cache,
    clear_weight_cache,
    lanczos_resize,
    set_cache_dir,
)


# --- Timing helper: CUDA events on GPU, perf_counter on CPU ---
def time_run(device: torch.device, fn):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        return out, ms
    else:
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000.0
        return out, ms


def tensor_to_pil(tensor):
    """Converts a (1, C, H, W) float tensor to a PIL Image."""
    # Ensure tensor is on CPU before converting to numpy
    image_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def create_label_image(lines, width, *, font_ratio=0.03125, font_name=None):
    """
    Create a label image whose font scales with the rendered panel width.
    font_ratio is the fraction of panel width used as font pixel height (≈3.125% by default).
    """
    # Choose a font; try common TrueType faces before falling back to PIL default.
    candidates = [font_name] if font_name else ["DejaVuSans.ttf", "arial.ttf"]
    font = None
    for face in candidates:
        try:
            if face:
                # Pixel size = proportional to panel width (no clamps to preserve invariance).
                font_size = max(1, int(round(width * float(font_ratio))))
                font = ImageFont.truetype(face, font_size)
                break
        except IOError:
            continue
    if font is None:
        # PIL default bitmap font (pixel-based); approximate size by drawing metrics
        font = ImageFont.load_default()
        # Default font is fixed; keep layout consistent without enforcing size here.

    # Compute line height from font metrics, add small padding between lines.
    ascent, descent = font.getmetrics()
    line_gap = max(2, int(round(font.size * 0.2))) if getattr(font, "size", None) else 4
    line_height = ascent + descent + line_gap

    total_height = (line_height * len(lines)) + 10
    img = Image.new("RGB", (width, total_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    y = 5
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (width - text_w) // 2
        draw.text((x, y), line, fill="white", font=font)

        # Optional underline for cache-hit line
        if line.strip().startswith("Cache Hit"):
            drawn_bbox = draw.textbbox((x, y), line, font=font)
            underline_y = drawn_bbox[3] + 2
            draw.line([(drawn_bbox[0], underline_y), (drawn_bbox[2], underline_y)], fill="red", width=2)

        y += line_height

    return img



def create_visual_comparison(
    original_pil,
    torchlanc_pil,
    pillow_pil,
    output_width,
    output_height,
    scale_factor,
    operation,
    *,
    font_ratio=0.03125,
):

    """Creates a 3-panel side-by-side image for visual quality comparison."""
    print(f"Creating visual comparison image for {operation}...")

    orig_w, orig_h = original_pil.size

    if operation == "downscale":
        # Panels rendered at ORIGINAL size
        panel_w, panel_h = orig_w, orig_h

        # Original: native size
        original_panel = original_pil

        # Downscaled results upscaled back to original with NEAREST for honest pixel comparison
        torchlanc_panel = torchlanc_pil.resize(
            (panel_w, panel_h), resample=Image.Resampling.NEAREST
        )
        pillow_panel = pillow_pil.resize(
            (panel_w, panel_h), resample=Image.Resampling.NEAREST
        )
    else:
        # 'upscale' behavior: show at target (upscaled) size
        panel_w, panel_h = output_width, output_height

        # Scale the original image using nearest-neighbor for honest comparison
        original_panel = original_pil.resize(
            (panel_w, panel_h), resample=Image.Resampling.NEAREST
        )
        torchlanc_panel = torchlanc_pil
        pillow_panel = pillow_pil

    # Create labels whose font scales with the actual rendered panel width
    label_original  = create_label_image(["Original"], panel_w, font_ratio=font_ratio)
    label_torchlanc = create_label_image([f"TorchLanc {scale_factor}x"], panel_w, font_ratio=font_ratio)
    label_pillow    = create_label_image([f"Pillow {scale_factor}x"], panel_w, font_ratio=font_ratio)
    label_height = label_original.height

    # Assemble the final image
    total_width = panel_w * 3
    total_height = panel_h + label_height
    comparison_img = Image.new("RGB", (total_width, total_height))

    # Paste labels and images
    comparison_img.paste(label_original, (0, 0))
    comparison_img.paste(original_panel, (0, label_height))

    comparison_img.paste(label_torchlanc, (panel_w, 0))
    comparison_img.paste(torchlanc_panel, (panel_w, label_height))

    comparison_img.paste(label_pillow, (panel_w * 2, 0))
    comparison_img.paste(pillow_panel, (panel_w * 2, label_height))

    # Draw separator lines
    draw = ImageDraw.Draw(comparison_img)
    for i in range(1, 3):
        x = panel_w * i
        draw.line([(x, 0), (x, total_height)], fill=(128, 128, 128), width=4)

    output_filename = f"comparison_visual_{operation}.png"
    comparison_img.save(output_filename)
    print(f"Saved visual comparison as {output_filename}.")


def run_race_test(device, test_config, cpu_name, gpu_name, *, font_ratio=0.03125):
    """Runs a complete benchmark for a given configuration and saves a comparison image."""

    # --- Setup from config ---
    image_filename = test_config["image_filename"]
    operation = test_config["operation"]
    batch_size = test_config["batch_size"]

    script_dir = os.path.dirname(__file__)

    print(f"\n--- RUNNING RACETEST: {operation.upper()} ON '{image_filename}' ---")

    # --- Load Data ---
    full_image_path = os.path.normpath(os.path.join(script_dir, image_filename))
    input_tensor_single = (
        torchvision.io.read_image(full_image_path).unsqueeze(0).float() / 255.0
    )
    input_batch_tensor = input_tensor_single.repeat(batch_size, 1, 1, 1).to(device)
    _, _, original_height, original_width = input_batch_tensor.shape

    # --- Determine Target Size ---
    if operation in ("downscale", "upscale"):
        scale_factor = test_config["scale_factor"]
        output_width = int(round(original_width * scale_factor))
        output_height = int(round(original_height * scale_factor))
    else:
        raise ValueError("Operation must be 'downscale' or 'upscale'")

    op_details = f"{operation.capitalize()}: {original_width}x{original_height} -> {output_width}x{output_height}"
    print(op_details)

    # --- Benchmark Our Resizer (GPU) ---
    print("Benchmarking TorchLanc GPU scaler...")
    # Run 1: Warm-up / Cache Miss / JIT Compile
    our_output_batch, duration_miss_ms = time_run(
        device,
        lambda: lanczos_resize(
            input_batch_tensor, height=output_height, width=output_width, a=3
        ),
    )
    # Run 2: Cached Run / Cache Hit
    _, duration_hit_ms = time_run(
        device,
        lambda: lanczos_resize(
            input_batch_tensor, height=output_height, width=output_width, a=3
        ),
    )

    # --- Benchmark Pillow Resizer (CPU) ---
    print(f"Benchmarking Pillow CPU Lanczos...")
    input_pil_list = [tensor_to_pil(input_tensor_single) for _ in range(batch_size)]
    pil_output_images = []

    start_time_pil = time.time()
    for img in input_pil_list:
        pil_output_images.append(
            img.resize((output_width, output_height), resample=Image.Resampling.LANCZOS)
        )
    end_time_pil = time.time()
    pil_duration_ms = (end_time_pil - start_time_pil) * 1000

    # --- Report Times ---
    print(f"\n--- RESULTS FOR BATCH SIZE: {batch_size} ({operation.upper()}) ---")
    print(f"TorchLanc (Cache Miss): {duration_miss_ms:.2f}ms total ({duration_miss_ms/batch_size:.2f}ms/img)")
    print(f"TorchLanc (Cache Hit):  {duration_hit_ms:.2f}ms total ({duration_hit_ms/batch_size:.2f}ms/img)")
    print(f"Pillow (CPU):           {pil_duration_ms:.2f}ms total ({pil_duration_ms/batch_size:.2f}ms/img)")

    # Relative vs TorchLanc cache-hit
    eps = 1e-9
    rel = (pil_duration_ms / max(duration_hit_ms, eps) - 1.0) * 100.0
    direction = "slower" if rel >= 0 else "faster"
    print(f"Pillow was {abs(rel):.1f}% {direction}")


    # --- Create and Save Benchmark Comparison Image ---
    print("\nCreating side-by-side benchmark image...")
    our_output_pil = tensor_to_pil(our_output_batch[0].unsqueeze(0))
    pil_output_image = pil_output_images[0]

    torchlanc_hit_ms_img = duration_hit_ms / batch_size
    pillow_ms_img = pil_duration_ms / batch_size
    torchlanc_title = f"TorchLanc ({device.type.upper()})"
    pillow_title = "Pillow (CPU)"

    our_label_lines = [
        torchlanc_title,
        f"Batch Size: {batch_size}",
        f"Cache Miss: {duration_miss_ms:.2f}ms total ({duration_miss_ms/batch_size:.2f}ms/img)",
        f"Cache Hit: {duration_hit_ms:.2f}ms total ({torchlanc_hit_ms_img:.2f}ms/img)",
    ]
    pil_label_lines = [
        pillow_title,
        f"Batch Size: {batch_size}",
        f"Total: {pil_duration_ms:.2f}ms ({pillow_ms_img:.2f}ms/img)",
        f"Pillow was {abs(rel):.1f}% {direction}",
    ]


    total_width = output_width * 2

    our_label = create_label_image(our_label_lines, width=output_width, font_ratio=font_ratio)
    pil_label = create_label_image(pil_label_lines, width=output_width, font_ratio=font_ratio)

    info_lines = [op_details, f"GPU: {gpu_name}", f"CPU: {cpu_name}"]
    info_label = create_label_image(info_lines, total_width, font_ratio=font_ratio)
    info_height = info_label.height


    label_height = our_label.height
    total_height_bench = output_height + label_height + info_height

    benchmark_img = Image.new(
        "RGB", (total_width, total_height_bench), color=(60, 60, 60)
    )

    benchmark_img.paste(info_label, (0, 0))
    benchmark_img.paste(our_label, (0, info_height))
    benchmark_img.paste(pil_label, (output_width, info_height))
    benchmark_img.paste(our_output_pil, (0, info_height + label_height))
    benchmark_img.paste(pil_output_image, (output_width, info_height + label_height))

    draw = ImageDraw.Draw(benchmark_img)
    separator_x = output_width
    separator_y_start = info_height + label_height
    separator_y_end = total_height_bench
    draw.line(
        [(separator_x, separator_y_start), (separator_x, separator_y_end)],
        fill=(128, 128, 128),
        width=4,
    )

    benchmark_filename = f"comparison_batch_{batch_size}_{operation}.png"
    benchmark_img.save(benchmark_filename)
    print(f"Saved benchmark image as {benchmark_filename}.")

    # --- Create and Save Visual Comparison Image ---
    create_visual_comparison(
        original_pil=tensor_to_pil(input_tensor_single),
        torchlanc_pil=our_output_pil,
        pillow_pil=pil_output_image,
        output_width=output_width,
        output_height=output_height,
        scale_factor=test_config["scale_factor"],
        operation=operation,
        font_ratio=font_ratio,
    )



def run_self_test(device, test_config):
    """Runs a benchmark of TorchLanc against itself, leveraging the cache."""
    # --- Setup from config ---
    image_filename = test_config["image_filename"]
    operation = test_config["operation"]
    batch_size = test_config["batch_size"]

    script_dir = os.path.dirname(__file__)  # Add this line

    # --- Load Data ---
    full_image_path = os.path.normpath(
        os.path.join(script_dir, image_filename)
    )  # Modify this line
    input_tensor_single = (
        torchvision.io.read_image(full_image_path).unsqueeze(0).float() / 255.0
    )
    input_batch_tensor = input_tensor_single.repeat(batch_size, 1, 1, 1).to(device)
    _, _, original_height, original_width = input_batch_tensor.shape

    # --- Determine Target Size ---
    if operation in ("downscale", "upscale"):
        scale_factor = test_config["scale_factor"]
        output_width = int(round(original_width * scale_factor))
        output_height = int(round(original_height * scale_factor))
    else:
        raise ValueError("Operation must be 'downscale' or 'upscale'")

    # --- Timed Run ---
    _, duration_ms = time_run(
        device,
        lambda: lanczos_resize(
            input_batch_tensor, height=output_height, width=output_width, a=3
        ),
    )

    print(
        f"  Batch Size: {batch_size:<5} | Time: {duration_ms:.2f}ms | Per Image: {duration_ms/batch_size:.2f}ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking tests for TorchLanc.")
    parser.add_argument("--self", action="store_true", help="Run TorchLanc self-benchmark using cache.")
    parser.add_argument("--race", action="store_true", help="Run TorchLanc vs. Pillow race.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size for --self (e.g., 256).")
    parser.add_argument("--op", choices=["downscale", "upscale"], default=None, help="Operation for --self override.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache dir to use (default: tests/.cache).")
    parser.add_argument("--font-ratio", type=float, default=0.03125, help="Font size as fraction of panel width (default 0.03125 ≈ 3.125%).")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if CUDA is available.")


    # Expand shorthands like: --self-256  or  --self-256-upscale
    argv = sys.argv[1:]
    expanded = []
    for tok in argv:
        m = re.match(r"--self-(\d+)(?:-(upscale|downscale))?$", tok)
        if m:
            expanded.extend(["--self", "--batch", m.group(1)])
            if m.group(2):
                expanded.extend(["--op", m.group(2)])
        else:
            expanded.append(tok)

    args = parser.parse_args(expanded)

    if not args.self and not args.race:
        parser.print_help()
        exit()


    # --- SYSTEM INFORMATION ---
    print("--- SYSTEM INFORMATION ---")
    device = torch.device("cpu" if args.cpu_only else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    gpu_name = "N/A"
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    cpu_name = "N/A"
    if cpuinfo:
        cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
        print(f"CPU: {cpu_name}")
    else:
        cpu_name = platform.processor()
        print(f"CPU: {cpu_name} (install py-cpuinfo for details)")


    # --- Test-local cache directory ---
    script_dir = os.path.dirname(__file__)
    cache_dir = args.cache_dir or os.path.join(script_dir, ".cache")
    set_cache_dir(cache_dir, reload_from_disk=True)

    # --- Test Configurations ---
    batch_sizes_race = [1, 8, 24, 48, 64]            # 128 removed (Pillow can crash)
    batch_sizes_self = [1, 8, 24, 48, 64, 128, 256]  # 256 added for stress

    downscale_config_base = {
        "image_filename": "test.png",
        "operation": "downscale",
        "scale_factor": 0.5,
    }

    upscale_config_base = {
        "image_filename": "test2.png",
        "operation": "upscale",
        "scale_factor": 2,
    }

    # --- Prerequisite Check ---
    required_files = [
        downscale_config_base["image_filename"],
        upscale_config_base["image_filename"],
    ]
    script_dir = os.path.dirname(__file__)
    for f in required_files:
        full_path = os.path.normpath(os.path.join(script_dir, f))
        if not os.path.exists(full_path):
            print(
                f"\nError: The test image '{full_path}' was not found. Please add it to this directory."
            )
            exit()

    if args.race:
        # --- Run Race Test Series ---
        for config_base in [downscale_config_base, upscale_config_base]:
            for batch in batch_sizes_race:
                # Clear caches via API (fresh miss timing; avoids filesystem races)
                clear_weight_cache(persist=True)
                clear_profile_cache(persist=True)
                set_cache_dir(cache_dir, reload_from_disk=True)

                current_config = config_base.copy()
                current_config["batch_size"] = batch
                run_race_test(device, current_config, cpu_name, gpu_name, font_ratio=args.font_ratio)


    if args.self:
        # --- Run Self-Test Series ---
        print("\n--- RUNNING TORCHLANC SELF-TEST ---")

        # Prime memory-profile cache if missing
        profile_path = os.path.join(cache_dir, "memory_profile.json")
        if not os.path.exists(profile_path):
            print("No memory profile found. Running a batch of 1 to create it...")
            clear_weight_cache(persist=True)
            clear_profile_cache(persist=True)
            set_cache_dir(cache_dir, reload_from_disk=True)

            cfg = downscale_config_base.copy()
            cfg["batch_size"] = 1
            run_self_test(device, cfg)

            cfg = upscale_config_base.copy()
            cfg["batch_size"] = 1
            run_self_test(device, cfg)
            print("Profile created.")

        # Determine which operations to run for --self
        ops = []
        if args.op is None:
            ops = [downscale_config_base, upscale_config_base]
        else:
            ops = [downscale_config_base if args.op == "downscale" else upscale_config_base]

        # Determine batches: override or default list
        batches = [args.batch] if args.batch is not None else batch_sizes_self

        for config_base in ops:
            print(f"\n--- BENCHMARKING {config_base['operation'].upper()} (CACHE-ASSISTED) ---")
            for batch in batches:
                current_config = config_base.copy()
                current_config["batch_size"] = int(batch)
                run_self_test(device, current_config)

    print("\n--- TESTING COMPLETE ---")
