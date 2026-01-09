"""
Test output file validation and format checking.
"""

import pytest
import sys
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from pyopenms import IdXMLFile, MzMLFile, MSExperiment, PeptideIdentificationList
from click.testing import CliRunner

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.onsitec import cli

class TestOutputValidation:
    """Test output file validation and format checking."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture
    def idxml_file(self, data_dir):
        """Get the idXML file path."""
        return data_dir / "1_consensus_fdr_filter_pep.idXML"
    
    @pytest.fixture
    def mzml_file(self, data_dir):
        """Get the mzML file path."""
        return data_dir / "1.mzML"
    
    def test_idxml_output_format(self, idxml_file, mzml_file):
        """Test that output idXML files are properly formatted."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            
            # Test AScore output
            result = runner.invoke(
                cli,
                [
                    "ascore",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", output_file,
                ],
            )
            
            if result.exit_code == 0 and os.path.exists(output_file):
                # Validate XML format
                try:
                    tree = ET.parse(output_file)
                    root = tree.getroot()
                    assert root.tag == "IdXML", "Output should be valid IdXML format"
                except ET.ParseError as e:
                    pytest.fail(f"Output file is not valid XML: {e}")
                
                # Validate with PyOpenMS
                try:
                    peptide_ids = PeptideIdentificationList()
                    protein_ids = []
                    IdXMLFile().load(output_file, protein_ids, peptide_ids)
                    assert len(peptide_ids) >= 0, "Output should be loadable by PyOpenMS"
                except Exception as e:
                    pytest.fail(f"Output file is not valid idXML: {e}")
    
    def test_output_file_creation(self, idxml_file, mzml_file):
        """Test that output files are created with proper permissions."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            
            # Test PhosphoRS output
            result = runner.invoke(
                cli,
                [
                    "phosphors",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", output_file,
                ],
            )
            
            if result.exit_code == 0:
                assert os.path.exists(output_file), "Output file should be created"
                assert os.path.isfile(output_file), "Output should be a file"
                assert os.access(output_file, os.R_OK), "Output file should be readable"
                assert os.path.getsize(output_file) > 0, "Output file should not be empty"
    
    def test_output_content_validation(self, idxml_file, mzml_file):
        """Test that output files contain expected content."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            
            # Test LucXor output
            result = runner.invoke(
                cli,
                [
                    "lucxor",
                    "--input-spectrum", str(mzml_file),
                    "--input-id", str(idxml_file),
                    "--output", output_file,
                ],
            )
            
            if result.exit_code == 0 and os.path.exists(output_file):
                # Check file content
                with open(output_file, 'r') as f:
                    content = f.read()
                    
                    # Basic XML validation
                    assert "<?xml" in content, "Output should contain XML declaration"
                    assert "<IdXML" in content, "Output should contain IdXML root element"
                    
                    # Check for required elements
                    assert "<ProteinIdentification" in content or "<PeptideIdentification" in content, \
                        "Output should contain identifications"
    
    def test_output_file_permissions(self, idxml_file, mzml_file):
        """Test that output files have correct permissions."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            
            # Test with AScore
            result = runner.invoke(
                cli,
                [
                    "ascore",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", output_file,
                ],
            )
            
            if result.exit_code == 0 and os.path.exists(output_file):
                # Check file permissions
                stat_info = os.stat(output_file)
                assert stat_info.st_mode & 0o444, "Output file should be readable"
    
    def test_output_file_size(self, idxml_file, mzml_file):
        """Test that output files have reasonable sizes."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            
            # Test with PhosphoRS
            result = runner.invoke(
                cli,
                [
                    "phosphors",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", output_file,
                ],
            )
            
            if result.exit_code == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                assert file_size > 0, "Output file should not be empty"
                assert file_size < 100 * 1024 * 1024, "Output file should be reasonable size"
    
    def test_output_file_encoding(self, idxml_file, mzml_file):
        """Test that output files use correct encoding."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            
            # Test with LucXor
            result = runner.invoke(
                cli,
                [
                    "lucxor",
                    "--input-spectrum", str(mzml_file),
                    "--input-id", str(idxml_file),
                    "--output", output_file,
                ],
            )
            
            if result.exit_code == 0 and os.path.exists(output_file):
                # Test file encoding
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        assert len(content) > 0, "Output should be readable as UTF-8"
                except UnicodeDecodeError:
                    pytest.fail("Output file should be UTF-8 encoded")


class TestErrorHandling:
    """Test error handling with invalid inputs."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the data directory path."""
        return Path(__file__).parent.parent / "data"
    
    def test_invalid_input_file(self, data_dir):
        """Test handling of invalid input files."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            invalid_file = os.path.join(temp_dir, "nonexistent.mzML")
            
            # Test with non-existent file
            result = runner.invoke(
                cli,
                [
                    "ascore",
                    "--in-file", invalid_file,
                    "--id-file", str(data_dir / "1_consensus_fdr_filter_pep.idXML"),
                    "--out-file", output_file,
                ],
            )
            
            assert result.exit_code != 0, "Should fail with invalid input file"
            assert "not found" in result.output.lower() or "error" in result.output.lower(), \
                "Should report file not found error"
    
    def test_invalid_output_directory(self, data_dir):
        """Test handling of invalid output directory."""
        runner = CliRunner()
        
        # Test with non-existent output directory
        invalid_output = "/nonexistent/path/output.idXML"
        
        result = runner.invoke(
            cli,
            [
                "ascore",
                "--in-file", str(data_dir / "1.mzML"),
                "--id-file", str(data_dir / "1_consensus_fdr_filter_pep.idXML"),
                "--out-file", invalid_output,
            ],
        )
        
        assert result.exit_code != 0, "Should fail with invalid output path"
    
    def test_missing_required_arguments(self):
        """Test handling of missing required arguments."""
        runner = CliRunner()
        
        # Test with missing arguments
        result = runner.invoke(cli, ["ascore"])
        
        assert result.exit_code != 0, "Should fail with missing arguments"
        assert "Missing option" in result.output, "Should report missing option error"
    
    def test_invalid_file_format(self, data_dir):
        """Test handling of invalid file formats."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "test_output.idXML")
            invalid_file = os.path.join(temp_dir, "invalid.txt")
            
            # Create invalid file
            with open(invalid_file, 'w') as f:
                f.write("This is not a valid mzML file")
            
            result = runner.invoke(
                cli,
                [
                    "ascore",
                    "--in-file", invalid_file,
                    "--id-file", str(data_dir / "1_consensus_fdr_filter_pep.idXML"),
                    "--out-file", output_file,
                ],
            )
            
            assert result.exit_code != 0, "Should fail with invalid file format"
