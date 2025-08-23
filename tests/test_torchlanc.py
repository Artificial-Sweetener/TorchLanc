import pytest
import torch
from torchlanc import lanczos_resize


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
