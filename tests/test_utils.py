import os
import pytest

from src.utils import setup_environment

@pytest.mark.parametrize("platform, mode, expected_env", [
    # AMD platform
    ("amd", "nogpu", {"JAX_PLATFORMS": "rocm", "ROCM_PATH": "/opt/rocm", "HIP_VISIBLE_DEVICES": "-1"}),
    ("amd", "1gpu0", {"JAX_PLATFORMS": "rocm", "ROCM_PATH": "/opt/rocm", "HIP_VISIBLE_DEVICES": "0"}),
    ("amd", "1gpu1", {"JAX_PLATFORMS": "rocm", "ROCM_PATH": "/opt/rocm", "HIP_VISIBLE_DEVICES": "1"}),
    ("amd", "2gpus", {"JAX_PLATFORMS": "rocm", "ROCM_PATH": "/opt/rocm", "HIP_VISIBLE_DEVICES": "0,1"}),
    ("amd", "all_gpus", {"JAX_PLATFORMS": "rocm", "ROCM_PATH": "/opt/rocm", "HIP_VISIBLE_DEVICES": "0,1"}),
    
    # NVIDIA platform
    ("nvidia", "nogpu", {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "-1"}),
    ("nvidia", "1gpu0", {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0"}),
    ("nvidia", "1gpu1", {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "1"}),
    ("nvidia", "2gpus", {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0,1"}),
    ("nvidia", "all_gpus", {"JAX_PLATFORMS": "cuda", "CUDA_VISIBLE_DEVICES": "0,1,2,3"}),
])
def test_setup_environment_valid(platform, mode, expected_env):
    """Test valid environment setup."""
    class Args:
        def __init__(self, platform, mode):
            self.platform = platform
            self.mode = mode

    setup_environment(Args(platform, mode))

    for key, value in expected_env.items():
        assert os.environ.get(key) == value, f"Expected {key}={value}, got {os.environ[key]}"

def test_setup_environment_invalid_platform():
    """Test invalid platform setup."""
    class Args:
        platform = "invalid"
        mode = "nogpu"

    with pytest.raises(ValueError, match="Invalid platform"):
        setup_environment(Args())

def test_setup_environment_invalid_mode():
    """Test invalid mode setup."""
    class Args:
        platform = "nvidia"
        mode = "invalid"

    with pytest.raises(ValueError, match="Invalid mode"):
        setup_environment(Args())

