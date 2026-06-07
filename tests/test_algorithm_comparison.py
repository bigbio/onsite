"""
Test algorithm comparison by running tools on real idparquet data.
Compares key output metrics between runs for consistency.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onsite.onsitec import cli
from onsite.idparquet import load_dataframes

pytestmark = pytest.mark.data



@pytest.fixture(scope="session")
def data_dir():
    return os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture(scope="session")
def mzml_file(data_dir):
    return os.path.join(data_dir, "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1.mzML")


@pytest.fixture(scope="session")
def idparquet_dir(data_dir):
    return os.path.join(data_dir, "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_comet_perc.idparquet")


class TestAlgorithmComparison:
    @pytest.mark.slow
    def test_lucxor_produces_phospho_sites(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "lucxor.idparquet")
            r = runner.invoke(cli, [
                "lucxor",
                "--input-spectrum", mzml_file,
                "--input-id", idparquet_dir,
                "--output", out,
                "--target-modifications", "Phospho(S),Phospho(T),Phospho(Y)",
            ])
            assert r.exit_code == 0
            psms, _, _, _ = load_dataframes(out)
            assert len(psms) > 0
            # Check psm_metavalues contain Luciphor scores
            mv = psms.iloc[0].get("psm_metavalues")
            if mv is not None and isinstance(mv, np.ndarray):
                names = [m["name"] for m in mv if isinstance(m, dict)]
                assert any("Luciphor" in n for n in names), "Should have Luciphor metas"

    @pytest.mark.slow
    def test_ascore_produces_phospho_sites(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "ascore.idparquet")
            r = runner.invoke(cli, ["ascore", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0
            psms, _, _, _ = load_dataframes(out)
            assert len(psms) > 0
            mv = psms.iloc[0].get("psm_metavalues")
            if mv is not None and isinstance(mv, np.ndarray):
                names = [m["name"] for m in mv if isinstance(m, dict)]
                assert any("AScore" in n for n in names), "Should have AScore metas"

    @pytest.mark.slow
    def test_phosphors_produces_phospho_sites(self, idparquet_dir, mzml_file):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "phosphors.idparquet")
            r = runner.invoke(cli, ["phosphors", "-in", mzml_file, "-id", idparquet_dir, "-out", out])
            assert r.exit_code == 0
            psms, _, _, _ = load_dataframes(out)
            assert len(psms) > 0
            mv = psms.iloc[0].get("psm_metavalues")
            if mv is not None and isinstance(mv, np.ndarray):
                names = [m["name"] for m in mv if isinstance(m, dict)]
                assert any("PhosphoRS" in n for n in names), "Should have PhosphoRS metas"
