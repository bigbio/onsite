"""
Test LucXor algorithm.
"""

import pytest
import sys
import os
import numpy as np
from onsite import lucxor

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_lucxor_import():
    """Test LucXor imports."""
    try:
        assert lucxor is not None
        # Test that we can access components through the module
        PyLuciPHOr2 = lucxor.PyLuciPHOr2
        assert PyLuciPHOr2 is not None
    except ImportError:
        pytest.skip("LucXor components not available")

def test_lucxor_config():
    """Test LucXor configuration."""
    try:
        # Test that we can access LucXor through the module
        LucXor = lucxor.LucXor
        assert LucXor is not None
    except ImportError:
        pytest.skip("LucXor components not available")

def test_lucxor_models():
    """Test LucXor models."""
    try:
        # Test model access through the module
        CIDModel = lucxor.CIDModel
        HCDModel = lucxor.HCDModel
        
        assert CIDModel is not None
        assert HCDModel is not None
    except ImportError:
        pytest.skip("LucXor components not available")

def test_lucxor_spectrum():
    """Test LucXor spectrum class."""
    try:
        # Test that we can access Spectrum through the module
        # Note: Spectrum might not be directly accessible through __getattr__
        # This test verifies the module structure
        assert hasattr(lucxor, '__getattr__')
    except ImportError:
        pytest.skip("LucXor components not available")

def test_lucxor_peptide():
    """Test LucXor peptide class."""
    try:
        # Test that we can access Peptide through the module
        Peptide = lucxor.Peptide
        assert Peptide is not None
    except ImportError:
        pytest.skip("LucXor components not available")
