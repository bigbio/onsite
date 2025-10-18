"""
Test imports for OnSite package.
"""

import pytest
import sys
import os
import onsite
from onsite import AScore
from onsite import calculate_phospho_localization_compomics_style
from onsite import lucxor

# Add the parent directory to the path so we can import onsite
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_onsite_import():
    """Test that the main onsite package can be imported."""
    
    assert hasattr(onsite, '__version__')
    assert onsite.__version__ == "0.0.1"

def test_ascore_import():
    """Test that AScore can be imported."""
    assert AScore is not None

def test_phosphors_import():
    """Test that PhosphoRS can be imported."""
    assert calculate_phospho_localization_compomics_style is not None

def test_lucxor_import():
    """Test that LucXor components can be imported."""
    try:
        assert lucxor is not None
        # Test that we can access PyLuciPHOr2 through the module
        PyLuciPHOr2 = lucxor.PyLuciPHOr2
        assert PyLuciPHOr2 is not None
    except ImportError:
        pytest.skip("LucXor components not available")

def test_cli_import():
    """Test that CLI can be imported."""
    from onsite import onsitec
    assert onsitec is not None
    assert hasattr(onsitec, 'main')
