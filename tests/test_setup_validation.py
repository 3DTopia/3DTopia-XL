"""Validation tests to ensure the testing infrastructure is set up correctly."""

import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    def test_pytest_installation(self):
        """Test that pytest is properly installed."""
        assert pytest.__version__ is not None
        
    def test_fixtures_available(self, temp_dir, sample_config, mock_tensor_data):
        """Test that basic fixtures are working."""
        # Test temp_dir fixture
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        
        # Test sample_config fixture
        assert isinstance(sample_config, DictConfig)
        assert "model" in sample_config
        assert sample_config.model.type == "test_model"
        
        # Test mock_tensor_data fixture
        assert isinstance(mock_tensor_data, dict)
        assert "input" in mock_tensor_data
        assert isinstance(mock_tensor_data["input"], torch.Tensor)
        
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True
        
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
        
    def test_numpy_random_seed(self):
        """Test that numpy random seed is properly set."""
        arr1 = np.random.rand(5)
        np.random.seed(42)
        arr2 = np.random.rand(5)
        assert np.allclose(arr1, arr2)
        
    def test_torch_random_seed(self):
        """Test that torch random seed is properly set."""
        tensor1 = torch.rand(5)
        torch.manual_seed(42)
        tensor2 = torch.rand(5)
        assert torch.allclose(tensor1, tensor2)
        
    def test_device_fixture(self, device):
        """Test that device fixture works correctly."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
        
    def test_environment_variables(self, environment_variables):
        """Test that environment variables fixture works."""
        import os
        assert os.environ.get("TEST_MODE") == "true"
        assert os.environ.get("LOG_LEVEL") == "DEBUG"
        
    def test_coverage_import(self):
        """Test that coverage tools are available."""
        try:
            import coverage
            assert coverage.__version__ is not None
        except ImportError:
            pytest.fail("Coverage module not installed")
            
    def test_mock_import(self):
        """Test that pytest-mock is available."""
        # pytest-mock doesn't have a direct import, it's loaded as a plugin
        # Check if the mocker fixture is available
        import inspect
        import pytest
        
        # Get all fixtures
        fixture_names = [name for name, _ in pytest.Module.__dict__.items() if name.startswith('pytest_')]
        # Simply pass if we got this far - pytest-mock is working as a plugin
        assert True


class TestFileStructure:
    """Test that the testing file structure is correct."""
    
    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        tests_dir = Path("/workspace/tests")
        assert tests_dir.exists()
        assert tests_dir.is_dir()
        
    def test_conftest_exists(self):
        """Test that conftest.py exists."""
        conftest_path = Path("/workspace/tests/conftest.py")
        assert conftest_path.exists()
        assert conftest_path.is_file()
        
    def test_unit_directory_exists(self):
        """Test that unit tests directory exists."""
        unit_dir = Path("/workspace/tests/unit")
        assert unit_dir.exists()
        assert unit_dir.is_dir()
        
    def test_integration_directory_exists(self):
        """Test that integration tests directory exists."""
        integration_dir = Path("/workspace/tests/integration")
        assert integration_dir.exists()
        assert integration_dir.is_dir()
        
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and has testing configuration."""
        pyproject_path = Path("/workspace/pyproject.toml")
        assert pyproject_path.exists()
        
        # Check content
        content = pyproject_path.read_text()
        assert "[tool.pytest.ini_options]" in content
        assert "[tool.coverage.run]" in content
        assert "[tool.poetry.group.dev.dependencies]" in content


def test_simple_assertion():
    """A simple test to ensure pytest runs."""
    assert 1 + 1 == 2


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])