import pytest
import torch

from torchlanc import lanczos_resize, clear_profile_cache, set_cache_dir
from torchlanc.torchlanc import _memory_profile_cache, _get_device_identifier, clear_weight_cache


@pytest.fixture
def checkerboard_tensor():
    """
    Creates a 1x1x4x4 tensor representing a simple checkerboard.
    B, C, H, W
    """
    tensor = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
    tensor[:, :, 0:2, 0:2] = 1.0
    tensor[:, :, 2:4, 2:4] = 1.0
    return tensor


def test_upscale(checkerboard_tensor):
    """
    Tests if upscaling to double the size produces the correct shape and dtype.
    """
    output = lanczos_resize(checkerboard_tensor, height=8, width=8)
    assert output.shape == (1, 1, 8, 8)
    assert output.dtype == checkerboard_tensor.dtype


def test_downscale(checkerboard_tensor):
    """
    Tests if downscaling to half the size produces the correct shape and dtype.
    """
    output = lanczos_resize(checkerboard_tensor, height=2, width=2)
    assert output.shape == (1, 1, 2, 2)
    assert output.dtype == checkerboard_tensor.dtype


def test_identity_resize(checkerboard_tensor):
    """
    Tests if resizing to the same dimensions results in a tensor of the same shape
    and approximately the same values.
    """
    output = lanczos_resize(checkerboard_tensor, height=4, width=4)
    assert output.shape == checkerboard_tensor.shape
    assert torch.allclose(output, checkerboard_tensor, atol=1e-6)


def test_invalid_input_shape():
    """
    Tests that a ValueError is raised for inputs that are not 4D tensors.
    """
    with pytest.raises(ValueError, match="Input must be a 4D tensor"):
        tensor_3d = torch.randn(3, 256, 256)
        lanczos_resize(tensor_3d, height=128, width=128)


def test_invalid_channel_count():
    """
    Tests that a ValueError is raised for unsupported channel counts (e.g., 2).
    """
    with pytest.raises(
        ValueError, match="Input must be a 4D tensor .* with 1, 3, or 4 channels"
    ):
        tensor_invalid_channels = torch.randn(1, 2, 32, 32)
        lanczos_resize(tensor_invalid_channels, height=64, width=64)


def test_non_floating_point_input():
    """
    Tests that a ValueError is raised for non-floating point input tensors.
    """
    with pytest.raises(ValueError, match="Input tensor must be floating point"):
        tensor_int = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
        lanczos_resize(tensor_int, height=64, width=64)


def test_alpha_channel_handling():
    """
    Tests that a 4-channel (RGBA) tensor is processed and returns a 4-channel tensor.
    """
    tensor_rgba = torch.rand(1, 4, 32, 32, dtype=torch.float32)
    output = lanczos_resize(tensor_rgba, height=64, width=64)
    assert output.shape == (1, 4, 64, 64)


def test_color_space_branch_differs(checkerboard_tensor):
    rgb = torch.linspace(0, 1, steps=16, dtype=torch.float32).reshape(1, 1, 4, 4)
    tensors = rgb.repeat(1, 3, 1, 1)
    out_linear = lanczos_resize(tensors, height=8, width=8, color_space="linear")
    out_srgb = lanczos_resize(tensors, height=8, width=8, color_space="srgb")
    assert torch.allclose(out_linear, out_srgb) is False


def test_chunk_size_auto_populates_cache(tmp_path, monkeypatch, checkerboard_tensor):
    cache_dir = tmp_path / "cache"
    set_cache_dir(str(cache_dir), reload_from_disk=True)
    clear_profile_cache(persist=False)
    _ = lanczos_resize(checkerboard_tensor, height=8, width=8, chunk_size=0)
    key = f"{_get_device_identifier()}::4::4::8::8::3::high::linear"
    assert key in _memory_profile_cache
    assert _memory_profile_cache[key]["max_safe_chunk"] > 0


def test_validate_range_env(monkeypatch):
    monkeypatch.setenv("TORCHLANC_VALIDATE_RANGE", "1")
    tensor = torch.ones(1, 3, 4, 4).to(torch.float32) * 2
    with pytest.raises(ValueError, match=r"values must be in \[0, 1\]"):
        lanczos_resize(tensor, height=2, width=2)
    monkeypatch.delenv("TORCHLANC_VALIDATE_RANGE", raising=False)


def test_alpha_rgb_isolation():
    rgb = torch.ones(1, 3, 8, 8)
    alpha = torch.linspace(0, 1, steps=64, dtype=torch.float32).reshape(1, 1, 8, 8)
    tensor = torch.cat([rgb, alpha], dim=1)
    out = lanczos_resize(tensor, height=4, width=4)
    rgb_out, alpha_out = out[:, :3], out[:, 3:]
    assert torch.allclose(rgb_out, torch.ones_like(rgb_out))
    assert alpha_out.min() >= 0 and alpha_out.max() <= 1


def test_cache_hit_matches_cold(checkerboard_tensor):
    first = lanczos_resize(checkerboard_tensor, height=8, width=8)
    from torchlanc import clear_weight_cache

    clear_weight_cache(persist=False)
    second = lanczos_resize(checkerboard_tensor, height=8, width=8)
    assert torch.allclose(first, second, atol=1e-6)
