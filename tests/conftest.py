"""
Test configuration and fixtures for OnSite tests.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def data_dir():
    """Get the data directory path."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def mzml_file(data_dir):
    """Get the mzML file path (rep1)."""
    return data_dir / "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1.mzML"


@pytest.fixture(scope="session")
def idparquet_dir(data_dir):
    """Get the idparquet directory path (rep1)."""
    return data_dir / "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_comet_perc.idparquet"


@pytest.fixture(scope="session")
def data_files_exist(data_dir, mzml_file, idparquet_dir):
    """Check that required data files exist."""
    if not data_dir.exists():
        pytest.skip("Data directory does not exist")
    if not mzml_file.exists():
        pytest.skip(f"mzML file not found: {mzml_file}")
    if not idparquet_dir.exists() or not (idparquet_dir / "psms.parquet").exists():
        pytest.skip(f"idparquet dir not found: {idparquet_dir}")
    return True


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_output_file(temp_output_dir):
    """Get a sample output file path."""
    return os.path.join(temp_output_dir, "test_output.idparquet")


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "data: marks tests that require data files")
    config.addinivalue_line("markers", "cli: marks tests that test CLI functionality")
    config.addinivalue_line("markers", "algorithm: marks tests that test algorithm functionality")


# Skip tests if required data files are missing
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests if data files are missing."""
    data_dir = Path(__file__).parent.parent / "data"
    mzml_file = data_dir / "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1.mzML"
    idparquet_dir = data_dir / "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_comet_perc.idparquet"

    if not data_dir.exists() or not mzml_file.exists() or not idparquet_dir.exists():
        for item in items:
            if "data" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Required data files not found (mzML or idparquet)"))
