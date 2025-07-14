"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> DictConfig:
    """Create a sample configuration for testing."""
    config = {
        "model": {
            "type": "test_model",
            "hidden_dim": 128,
            "num_layers": 4,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 10,
        },
        "data": {
            "dataset": "test_dataset",
            "num_workers": 4,
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def mock_tensor_data() -> Dict[str, torch.Tensor]:
    """Create mock tensor data for testing."""
    return {
        "input": torch.randn(4, 3, 256, 256),
        "target": torch.randn(4, 128, 128, 128),
        "mask": torch.ones(4, 1, 256, 256),
    }


@pytest.fixture
def mock_numpy_data() -> Dict[str, np.ndarray]:
    """Create mock numpy data for testing."""
    return {
        "points": np.random.randn(1000, 3).astype(np.float32),
        "colors": np.random.randint(0, 255, (1000, 3), dtype=np.uint8),
        "normals": np.random.randn(1000, 3).astype(np.float32),
    }


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a sample image file for testing."""
    image_path = temp_dir / "test_image.png"
    # Create a simple 10x10 white image
    import numpy as np
    from PIL import Image
    
    img_array = np.ones((10, 10, 3), dtype=np.uint8) * 255
    img = Image.fromarray(img_array)
    img.save(image_path)
    return image_path


@pytest.fixture
def sample_mesh_data() -> Dict[str, Any]:
    """Create sample mesh data for testing."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ], dtype=np.int32)
    
    return {
        "vertices": vertices,
        "faces": faces,
        "vertex_colors": np.random.rand(4, 3).astype(np.float32),
    }


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@pytest.fixture
def mock_model_weights(temp_dir: Path) -> Path:
    """Create mock model weights file."""
    weights_path = temp_dir / "model_weights.pth"
    torch.save({
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"param_groups": []},
        "epoch": 5,
        "loss": 0.123,
    }, weights_path)
    return weights_path


@pytest.fixture
def environment_variables() -> Generator[Dict[str, str], None, None]:
    """Temporarily set environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        "TEST_MODE": "true",
        "LOG_LEVEL": "DEBUG",
    }
    os.environ.update(test_env)
    yield test_env
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Custom markers configuration
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


# Hook to skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests when appropriate."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)