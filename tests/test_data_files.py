"""
Test data file processing with real idparquet and mzML files.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.idparquet import load_dataframes
from onsite.onsitec import cli

pytestmark = pytest.mark.data



@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def idparquet_dir(data_dir):
    return os.path.join(data_dir, "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_comet_perc.idparquet")


@pytest.fixture
def mzml_file(data_dir):
    return os.path.join(data_dir, "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1.mzML")


class TestDataFileLoading:
    def test_data_files_exist(self, data_dir, idparquet_dir, mzml_file):
        assert os.path.isdir(data_dir), "Data directory should exist"
        assert os.path.isdir(idparquet_dir), "idparquet directory should exist"
        assert os.path.isfile(os.path.join(idparquet_dir, "psms.parquet")), "psms.parquet should exist"
        assert os.path.isfile(mzml_file), "mzML file should exist"

    def test_idparquet_loading(self, idparquet_dir):
        psms_df, proteins_df, _, _ = load_dataframes(idparquet_dir)
        assert len(psms_df) > 0, "Should load PSMs"
        assert "peptidoform" in psms_df.columns, "peptidoform column should exist"
        assert "peptide_identification_index" in psms_df.columns
        assert "hit_index" in psms_df.columns

    def test_phospho_peptides_present(self, idparquet_dir):
        psms_df, _, _, _ = load_dataframes(idparquet_dir)
        phospho = psms_df[psms_df["peptidoform"].str.contains("UNIMOD:21", na=False)]
        assert len(phospho) > 0, "Should have phosphorylated peptides"

    def test_psm_metavalues(self, idparquet_dir):
        psms_df, _, _, _ = load_dataframes(idparquet_dir)
        assert "psm_metavalues" in psms_df.columns
        row0 = psms_df.iloc[0]
        mv = row0["psm_metavalues"]
        assert isinstance(mv, np.ndarray), "psm_metavalues should be numpy array"

    def test_proteins_loading(self, idparquet_dir):
        _, proteins_df, _, _ = load_dataframes(idparquet_dir)
        assert len(proteins_df) > 0, "Should load proteins"
        assert "accession" in proteins_df.columns


class TestCLIWithRealData:
    def test_ascore_cli_with_real_data(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "ascore_result.idparquet")
            r = runner.invoke(cli, ["ascore", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0, f"AScore CLI should succeed: {r.output[:200]}"
            psms, _, _, _ = load_dataframes(out)
            assert len(psms) > 0, "Should produce output PSMs"

    def test_phosphors_cli_with_real_data(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "phosphors_result.idparquet")
            r = runner.invoke(cli, ["phosphors", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0, f"PhosphoRS CLI should succeed: {r.output[:200]}"
            psms, _, _, _ = load_dataframes(out)
            assert len(psms) > 0, "Should produce output PSMs"

    def test_lucxor_cli_with_real_data(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "lucxor_result.idparquet")
            r = runner.invoke(cli, [
                "lucxor",
                "--input-spectrum", mzml_file,
                "--input-id", idparquet_dir,
                "--output", out,
                "--target-modifications", "Phospho(S),Phospho(T),Phospho(Y)",
            ])
            assert r.exit_code == 0, f"LucXor CLI should succeed: {r.output[:200]}"
            psms, _, _, _ = load_dataframes(out)
            assert len(psms) > 0, "Should produce output PSMs"
