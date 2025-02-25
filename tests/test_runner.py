import pytest

from unittest.mock import patch
from src.runner import validate_arguments

@pytest.fixture
def test_validate_arguments():
    class Args:
        def __init__(self, input_fasta, preset, mode, platform, msa):
            self.input_fasta = input_fasta
            self.mode = mode
            self.preset = preset
            self.platform = platform
            self.msa = msa

    args = Args("input.fasta", "nogpu", "monomer", "nvidia", "msa")
    with patch("os.path.isfile", return_value=True):
        assert validate_arguments(args)

def test_validate_arguments_invalid_file():
    """Test argument validation with a missing input file."""
    class Args:
        input_fasta = "nonexistent.fasta"
        preset = "monomer"
        mode = "nogpu"
        platform = "nvidia"
        msa = "msa"
    
    with pytest.raises(FileNotFoundError):
        validate_arguments(Args())

def test_validate_arguments_invalid_preset():
    """Test argument validation with an invalid preset."""
    class Args:
        input_fasta = "input.fasta"
        preset = "invalid"
        mode = "nogpu"
        platform = "nvidia"
        msa = "msa"
    
    with patch("os.path.isfile", return_value=True):
        with pytest.raises(ValueError, match="Invalid preset"):
            validate_arguments(Args())

def test_validate_arguments_invalid_platform():
    """Test argument validation with an invalid platform."""
    class Args:
        input_fasta = "input.fasta"
        preset = "monomer"
        mode = "nogpu"
        platform = "invalid"
        msa = "msa"
    
    with patch("os.path.isfile", return_value=True):
        with pytest.raises(ValueError, match="Invalid platform"):
            validate_arguments(Args())

def test_validate_arguments_invalid_mode():
    """Test argument validation with an invalid mode."""
    class Args:
        input_fasta = "input.fasta"
        preset = "monomer"
        mode = "invalid"
        platform = "nvidia"
        msa = "msa"
    
    with patch("os.path.isfile", return_value=True):
        with pytest.raises(ValueError, match="Invalid mode"):
            validate_arguments(Args())
