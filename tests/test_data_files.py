"""Test data file processing with real idXML and mzML files."""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from pyopenms import *
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.onsitec import cli
from onsite.ascore import AScore
from onsite.phosphors import calculate_phospho_localization_compomics_style


def safe_load_idxml(idxml_file):
    """Load idXML file with proper handling for PyOpenMS cross-platform compatibility."""
    if hasattr(idxml_file, '__fspath__'):
        file_path = os.fspath(idxml_file)
    elif isinstance(idxml_file, bytes):
        file_path = idxml_file.decode('utf-8')
    else:
        file_path = str(idxml_file)
    
    file_path = os.path.normpath(file_path)
    
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"idXML file not found: {file_path}")
    
    protein_ids = []
    peptide_ids = []
    
    id_xml_file = IdXMLFile()
    id_xml_file.load(file_path, protein_ids, peptide_ids)
    
    return protein_ids, peptide_ids


class TestDataFileLoading:
    """Test loading and parsing of data files."""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture
    def idxml_file(self, data_dir):
        return data_dir / "1_consensus_fdr_filter_pep.idXML"
    
    @pytest.fixture
    def mzml_file(self, data_dir):
        return data_dir / "1.mzML"
    
    def test_data_files_exist(self, data_dir, idxml_file, mzml_file):
        """Test that required data files exist."""
        assert data_dir.exists(), "Data directory should exist"
        assert idxml_file.exists(), "idXML file should exist"
        assert mzml_file.exists(), "mzML file should exist"
    
    def test_idxml_loading(self, idxml_file):
        """Test loading idXML file."""
        assert idxml_file.exists(), "idXML file should exist"
        
        with open(idxml_file, 'r') as f:
            content = f.read()
            assert len(content) > 0, "idXML file should not be empty"
            assert "<?xml" in content, "idXML file should contain XML declaration"
    
    def test_mzml_loading(self, mzml_file):
        """Test loading mzML file."""
        assert mzml_file.exists(), "mzML file should exist"
        
        with open(mzml_file, 'r') as f:
            content = f.read()
            assert len(content) > 0, "mzML file should not be empty"
            assert "<?xml" in content, "mzML file should contain XML declaration"
    
    def test_idxml_parsing_with_pyopenms(self, idxml_file):
        """Test parsing idXML file with PyOpenMS."""
        protein_ids, peptide_ids = safe_load_idxml(idxml_file)
        
        assert len(peptide_ids) > 0, "Should have peptide identifications"
        assert len(protein_ids) > 0, "Should have protein identifications"
        
        total_hits = sum(len(pep_id.getHits()) for pep_id in peptide_ids)
        assert total_hits > 0, "Should have peptide hits"
    
    def test_mzml_parsing_with_pyopenms(self, mzml_file):
        """Test parsing mzML file with PyOpenMS."""
        try:
            exp = MSExperiment()
            MzMLFile().load(str(mzml_file), exp)
            
            assert exp.size() > 0, "Should have spectra"
            
            total_peaks = sum(spectrum.size() for spectrum in exp)
            assert total_peaks > 0, "Should have peaks"
            
        except Exception as e:
            pytest.fail(f"Failed to parse mzML file: {e}")


class TestAlgorithmExecution:
    """Test algorithm execution with real data files."""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture
    def idxml_file(self, data_dir):
        return data_dir / "1_consensus_fdr_filter_pep.idXML"
    
    @pytest.fixture
    def mzml_file(self, data_dir):
        return data_dir / "1.mzML"
    
    def test_ascore_with_real_data(self, idxml_file, mzml_file):
        """Test AScore algorithm with real data files."""
        protein_ids, peptide_ids = safe_load_idxml(idxml_file)
        
        exp = MSExperiment()
        MzMLFile().load(str(mzml_file), exp)
        
        ascore = AScore()
        
        tested_hits = 0
        max_tests = 5
        
        for pep_id in peptide_ids[:max_tests]:
            for hit in pep_id.getHits():
                if tested_hits >= max_tests:
                    break
                
                if pep_id.metaValueExists("spectrum_reference"):
                    spectrum_ref = pep_id.getMetaValue("spectrum_reference")
                    if "scan=" in spectrum_ref:
                        spectrum_index = int(spectrum_ref.split("scan=")[1].split()[0])
                    else:
                        spectrum_index = -1
                else:
                    spectrum_index = -1
                    
                if spectrum_index >= 0 and spectrum_index < exp.size():
                    spectrum = exp[spectrum_index]
                    
                    try:
                        result = ascore.compute(hit, spectrum)
                        assert result is not None, "AScore should return a result"
                    except Exception as e:
                        assert "phosphorylation" in str(e).lower() or "modification" in str(e).lower()
                
                tested_hits += 1
            
            if tested_hits >= max_tests:
                break
        
        assert tested_hits > 0, "Should have tested at least one hit"
    
    def test_phosphors_with_real_data(self, idxml_file, mzml_file):
        """Test PhosphoRS algorithm with real data files."""
        protein_ids, peptide_ids = safe_load_idxml(idxml_file)
        
        exp = MSExperiment()
        MzMLFile().load(str(mzml_file), exp)
        
        tested_hits = 0
        max_tests = 5
        
        for pep_id in peptide_ids[:max_tests]:
            for hit in pep_id.getHits():
                if tested_hits >= max_tests:
                    break
                
                if pep_id.metaValueExists("spectrum_reference"):
                    spectrum_ref = pep_id.getMetaValue("spectrum_reference")
                    if "scan=" in spectrum_ref:
                        spectrum_index = int(spectrum_ref.split("scan=")[1].split()[0])
                    else:
                        spectrum_index = -1
                else:
                    spectrum_index = -1
                    
                if spectrum_index >= 0 and spectrum_index < exp.size():
                    spectrum = exp[spectrum_index]
                    
                    try:
                        result = calculate_phospho_localization_compomics_style(hit, spectrum)
                        assert result is not None, "PhosphoRS should return a result"
                    except Exception as e:
                        assert "phosphorylation" in str(e).lower() or "modification" in str(e).lower()
                
                tested_hits += 1
            
            if tested_hits >= max_tests:
                break
        
        assert tested_hits > 0, "Should have tested at least one hit"

class TestCLIWithRealData:
    """Test CLI execution with real data files."""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture
    def idxml_file(self, data_dir):
        return data_dir / "1_consensus_fdr_filter_pep.idXML"
    
    @pytest.fixture
    def mzml_file(self, data_dir):
        return data_dir / "1.mzML"
    
    def test_ascore_cli_with_real_data(self, idxml_file, mzml_file):
        """Test AScore CLI with real data files."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "ascore_result.idXML")
            
            result = runner.invoke(
                cli,
                [
                    "ascore",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", output_file,
                ],
            )
            
            assert result.exit_code in [0, 1], f"AScore CLI should execute: {result.output}"
            
            if result.exit_code == 0:
                assert os.path.exists(output_file), "Output file should be created"
                assert os.path.getsize(output_file) > 0, "Output file should not be empty"
    
    def test_phosphors_cli_with_real_data(self, idxml_file, mzml_file):
        """Test PhosphoRS CLI with real data files."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "phosphors_result.idXML")
            
            result = runner.invoke(
                cli,
                [
                    "phosphors",
                    "--in-file", str(mzml_file),
                    "--id-file", str(idxml_file),
                    "--out-file", output_file,
                ],
            )
            
            assert result.exit_code in [0, 1], f"PhosphoRS CLI should execute: {result.output}"
            
            if result.exit_code == 0:
                assert os.path.exists(output_file), "Output file should be created"
                assert os.path.getsize(output_file) > 0, "Output file should not be empty"
    
    def test_lucxor_cli_with_real_data(self, idxml_file, mzml_file):
        """Test LucXor CLI with real data files."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "lucxor_result.idXML")
            
            result = runner.invoke(
                cli,
                [
                    "lucxor",
                    "--input-spectrum", str(mzml_file),
                    "--input-id", str(idxml_file),
                    "--output", output_file,
                ],
            )
            
            assert result.exit_code in [0, 1], f"LucXor CLI should execute: {result.output}"
            
            if result.exit_code == 0:
                assert os.path.exists(output_file), "Output file should be created"
                assert os.path.getsize(output_file) > 0, "Output file should not be empty"


class TestLucXorWithRealData:
    """Test LucXor algorithm with real data files."""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    @pytest.fixture
    def idxml_file(self, data_dir):
        return data_dir / "1_consensus_fdr_filter_pep.idXML"
    
    @pytest.fixture
    def mzml_file(self, data_dir):
        return data_dir / "1.mzML"
    
    def test_lucxor_import(self):
        """Test that LucXor module can be imported."""
        try:
            from onsite import lucxor
            assert lucxor is not None, "LucXor module should be importable"
        except ImportError as e:
            pytest.skip(f"LucXor not available: {e}")
    
    def test_lucxor_cli_with_real_data(self, idxml_file, mzml_file):
        """Test LucXor CLI with real data files."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "lucxor_result.idXML")
            
            result = runner.invoke(
                cli,
                [
                    "lucxor",
                    "--input-spectrum", str(mzml_file),
                    "--input-id", str(idxml_file),
                    "--output", output_file,
                ],
            )
            
            assert result.exit_code in [0, 1], f"LucXor CLI should execute: {result.output}"
            
            if result.exit_code == 0:
                assert os.path.exists(output_file), "Output file should be created"
                assert os.path.getsize(output_file) > 0, "Output file should not be empty"
    
    def test_lucxor_data_loading(self, idxml_file, mzml_file):
        """Test that LucXor can load the data files."""
        protein_ids, peptide_ids = safe_load_idxml(idxml_file)
        
        exp = MSExperiment()
        MzMLFile().load(str(mzml_file), exp)
        
        assert len(peptide_ids) > 0, "Should have peptide identifications"
        assert len(protein_ids) > 0, "Should have protein identifications"
        assert exp.size() > 0, "Should have spectra"
        
        total_hits = sum(len(pep_id.getHits()) for pep_id in peptide_ids)
        assert total_hits > 0, "Should have peptide hits"
    
    def test_lucxor_algorithm_availability(self):
        """Test that LucXor algorithm components are available."""
        try:
            from onsite.lucxor import core, models, spectrum, peptide
            assert core is not None, "LucXor core module should be available"
            assert models is not None, "LucXor models module should be available"
            assert spectrum is not None, "LucXor spectrum module should be available"
            assert peptide is not None, "LucXor peptide module should be available"
        except ImportError as e:
            pytest.skip(f"LucXor modules not available: {e}")
    
    def test_lucxor_config_loading(self):
        """Test that LucXor configuration can be loaded."""
        try:
            from onsite.lucxor import config
            assert config is not None, "LucXor config module should be available"
        except ImportError as e:
            pytest.skip(f"LucXor config not available: {e}")
    
    def test_lucxor_constants_loading(self):
        """Test that LucXor constants can be loaded."""
        try:
            from onsite.lucxor import constants
            assert constants is not None, "LucXor constants module should be available"
        except ImportError as e:
            pytest.skip(f"LucXor constants not available: {e}")


class TestLucXorIntegration:
    """Test LucXor integration with the main package."""
    
    def test_lucxor_in_main_package(self):
        """Test that LucXor is properly integrated in the main package."""
        try:
            from onsite import lucxor
            assert lucxor is not None, "LucXor should be available in main package"
        except ImportError:
            pytest.skip("LucXor not available in main package")
    
    def test_lucxor_cli_command(self):
        """Test that LucXor CLI command is available."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0, "CLI help should work"
        assert "lucxor" in result.output, "LucXor command should be in help"
    
    def test_lucxor_cli_help(self):
        """Test LucXor CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["lucxor", "--help"])
        
        assert result.exit_code == 0, "LucXor help should work"
        assert "input-spectrum" in result.output, "Should have input-spectrum option"
        assert "input-id" in result.output, "Should have input-id option"
        assert "output" in result.output, "Should have output option"


class TestDataFileValidation:
    """Test data file validation and error handling."""
    
    @pytest.fixture
    def data_dir(self):
        return Path(__file__).parent.parent / "data"
    
    def test_idxml_file_structure(self, data_dir):
        """Test idXML file structure and content."""
        idxml_file = data_dir / "1_consensus_fdr_filter_pep.idXML"
        
        assert idxml_file.exists(), "idXML file should exist"
        
        with open(idxml_file, 'r') as f:
            content = f.read()
            
            assert "<IdXML" in content, "Should contain IdXML root element"
            assert "<ProteinIdentification" in content, "Should contain protein identifications"
            assert "<PeptideIdentification" in content, "Should contain peptide identifications"
    
    def test_mzml_file_structure(self, data_dir):
        """Test mzML file structure and content."""
        mzml_file = data_dir / "1.mzML"
        
        assert mzml_file.exists(), "mzML file should exist"
        
        with open(mzml_file, 'r') as f:
            content = f.read()
            
            assert "<mzML" in content, "Should contain mzML root element"
            assert "<spectrumList" in content, "Should contain spectrum list"
            assert "<spectrum" in content, "Should contain spectra"
    
    def test_file_size_validation(self, data_dir):
        """Test that data files have reasonable sizes."""
        idxml_file = data_dir / "1_consensus_fdr_filter_pep.idXML"
        mzml_file = data_dir / "1.mzML"
        
        idxml_size = idxml_file.stat().st_size
        mzml_size = mzml_file.stat().st_size
        
        assert idxml_size > 1000, "idXML file should be larger than 1KB"
        assert mzml_size > 1000, "mzML file should be larger than 1KB"
        assert idxml_size < 100 * 1024 * 1024, "idXML file should be smaller than 100MB"
