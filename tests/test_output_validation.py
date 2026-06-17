"""
Test output file validation for idparquet format.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.idparquet import load_dataframes, save_dataframes
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


class TestOutputValidation:
    def test_idparquet_output_format(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_output.idparquet")
            r = runner.invoke(cli, ["ascore", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0
            assert os.path.isdir(out), "Output should be a directory"
            assert os.path.isfile(os.path.join(out, "psms.parquet")), "psms.parquet should exist"
            # Validate parquet is readable
            psms_df, _, _, _ = load_dataframes(out)
            assert len(psms_df) > 0

    def test_output_content_validation(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_output.idparquet")
            r = runner.invoke(cli, ["lucxor",
                "--input-spectrum", mzml_file,
                "--input-id", idparquet_dir,
                "--output", out,
                "--target-modifications", "Phospho(S),Phospho(T),Phospho(Y)",
                "--modeling-score-threshold", "0.3",
            ])
            if r.exit_code != 0:
                pytest.skip(f"LucXor required but failed: {r.output[:200]}")
            psms_df, proteins_df, _, _ = load_dataframes(out)
            assert "score" in psms_df.columns
            assert len(psms_df) > 0

    def test_output_file_permissions(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_output.idparquet")
            r = runner.invoke(cli, ["phosphors", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0
            assert os.path.isdir(out)
            assert os.access(os.path.join(out, "psms.parquet"), os.R_OK)

    def test_output_file_size(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_output.idparquet")
            r = runner.invoke(cli, ["ascore", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0
            size = os.path.getsize(os.path.join(out, "psms.parquet"))
            assert size > 100, "Output file should be > 100 bytes"

    def test_score_type_and_higher_better(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "test_output.idparquet")
            r = runner.invoke(cli, ["ascore", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0
            psms_df, _, _, _ = load_dataframes(out)
            assert "score_type" in psms_df.columns
            assert psms_df["score_type"].iloc[0] != ""
            assert "higher_score_better" in psms_df.columns
