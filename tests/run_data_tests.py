#!/usr/bin/env python3
"""
Test runner for data file tests.

This script runs all tests related to data file processing and validation.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_data_tests():
    """Run all data file tests."""
    print("Running data file tests...")
    
    # Test files to run
    test_files = [
        "test_data_files.py",
        "test_output_validation.py"
    ]
    
    # Run each test file
    for test_file in test_files:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            print(f"\nRunning {test_file}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", str(test_path), "-v"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"✅ {test_file} passed")
                else:
                    print(f"❌ {test_file} failed")
                    print(result.stdout)
                    print(result.stderr)
            except Exception as e:
                print(f"❌ Error running {test_file}: {e}")
        else:
            print(f"⚠️  {test_file} not found")


def run_specific_algorithm_tests():
    """Run tests for specific algorithms."""
    algorithms = ["ascore", "phosphors", "lucxor"]
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm} tests...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                f"test_data_files.py::TestAlgorithmExecution::test_{algorithm}_with_real_data",
                "-v"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {algorithm} tests passed")
            else:
                print(f"❌ {algorithm} tests failed")
                print(result.stdout)
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error running {algorithm} tests: {e}")


def run_cli_tests():
    """Run CLI tests with real data."""
    print("\nRunning CLI tests with real data...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "test_data_files.py::TestCLIWithRealData",
            "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CLI tests passed")
        else:
            print("❌ CLI tests failed")
            print(result.stdout)
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error running CLI tests: {e}")


def main():
    """Main test runner."""
    print("OnSite Data File Test Runner")
    print("=" * 40)
    
    # Check if data files exist
    data_dir = Path(__file__).parent.parent / "data"
    idparquet_dir = data_dir / "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_feat_perc.idparquet"
    mzml_file = data_dir / "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1.mzML"

    if not data_dir.exists():
        print("Data directory not found")
        return 1

    if not idparquet_dir.exists() or not (idparquet_dir / "psms.parquet").exists():
        print("idparquet directory not found")
        return 1

    if not mzml_file.exists():
        print("mzML file not found")
        return 1
    
    print("✅ Data files found")
    
    # Run tests
    run_data_tests()
    run_specific_algorithm_tests()
    run_cli_tests()
    
    print("\n" + "=" * 40)
    print("Test run completed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
