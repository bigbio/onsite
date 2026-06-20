"""
Tests for onsite.id_io — Phase 1: load side.

Covers:
  - detect_format: idparquet, idxml, mzid, ValueError on unknown
  - load_identifications idParquet native: 4-tuple, >0 rows, required columns
  - load_identifications idXML: 2 hits, peptidoform/charge/scan/score, MetaValues
  - load_identifications mzid:  same as idxml but via MzIdentMLFile
"""

import os
import re

import numpy as np
import pytest

from onsite.id_io import detect_format, load_identifications

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_FIXTURE_IDPARQUET = os.path.join(
    _DATA,
    "SF_200217_pPeptideLibrary_pool1_HCDnlETcaD_OT_rep1_comet_perc.idparquet",
)


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_directory_is_idparquet(self, tmp_path):
        d = tmp_path / "foo"
        d.mkdir()
        assert detect_format(str(d)) == "idparquet"

    def test_dotidparquet_extension(self, tmp_path):
        p = tmp_path / "foo.idparquet"
        # does not need to exist as a dir – extension alone is enough
        assert detect_format(str(p)) == "idparquet"

    def test_idxml(self, tmp_path):
        p = tmp_path / "foo.idXML"
        assert detect_format(str(p)) == "idxml"

    def test_mzid(self, tmp_path):
        p = tmp_path / "foo.mzid"
        assert detect_format(str(p)) == "mzid"

    def test_mzidentml(self, tmp_path):
        p = tmp_path / "foo.mzIdentML"
        assert detect_format(str(p)) == "mzid"

    def test_unknown_raises(self, tmp_path):
        p = tmp_path / "foo.xyz"
        with pytest.raises(ValueError, match="foo.xyz"):
            detect_format(str(p))


# ---------------------------------------------------------------------------
# load_identifications — idParquet native
# ---------------------------------------------------------------------------


class TestLoadIdparquetNative:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not os.path.isdir(_FIXTURE_IDPARQUET):
            pytest.skip(f"Fixture not found: {_FIXTURE_IDPARQUET}")

    def test_returns_four_dataframes(self):
        result = load_identifications(_FIXTURE_IDPARQUET)
        assert len(result) == 4, "Expected 4-tuple (psms, proteins, search_params, protein_groups)"

    def test_psms_has_rows(self):
        psms_df, *_ = load_identifications(_FIXTURE_IDPARQUET)
        assert len(psms_df) > 0, "psms_df must have at least one row"

    def test_psms_has_required_columns(self):
        psms_df, *_ = load_identifications(_FIXTURE_IDPARQUET)
        for col in ("peptidoform", "score", "psm_metavalues"):
            assert col in psms_df.columns, f"Missing column: {col}"

    def test_psm_metavalues_structure(self):
        psms_df, *_ = load_identifications(_FIXTURE_IDPARQUET)
        mv = psms_df["psm_metavalues"].iloc[0]
        assert isinstance(mv, np.ndarray)
        entry = mv[0]
        assert "name" in entry and "value" in entry and "value_type" in entry


# ---------------------------------------------------------------------------
# Helpers — build minimal idXML / mzid fixtures
# ---------------------------------------------------------------------------


def _build_pep_list():
    """Return (prot_list, PeptideIdentificationList) with 2 hits in 1 PID."""
    import pyopenms as oms

    pi = oms.ProteinIdentification()
    pi.setIdentifier("run1")
    pi.setScoreType("Percolator_qvalue")
    pi.setHigherScoreBetter(False)
    sp = pi.getSearchParameters()
    sp.db = "test_db"
    pi.setSearchParameters(sp)
    prot = [pi]

    pep_list = oms.PeptideIdentificationList()

    pid = oms.PeptideIdentification()
    pid.setScoreType("Percolator_qvalue")
    pid.setHigherScoreBetter(False)
    pid.setRT(100.5)
    pid.setMZ(500.0)
    pid.setIdentifier("run1")
    pid.setMetaValue("spectrum_reference", "controllerType=0 controllerNumber=1 scan=5")

    # hit 0 — phospho on S
    hit0 = oms.PeptideHit()
    hit0.setSequence(oms.AASequence.fromString("S(Phospho)PEK"))
    hit0.setCharge(2)
    hit0.setScore(0.01)
    hit0.setMetaValue("target_decoy", "target")
    hit0.setMetaValue("AScore_site_scores", "{1: 50.0}")

    # hit 1 — unmodified peptide
    hit1 = oms.PeptideHit()
    hit1.setSequence(oms.AASequence.fromString("PEK"))
    hit1.setCharge(3)
    hit1.setScore(0.05)
    hit1.setMetaValue("target_decoy", "decoy")

    pid.setHits([hit0, hit1])
    pep_list.push_back(pid)

    return prot, pep_list


# ---------------------------------------------------------------------------
# load_identifications — idXML
# ---------------------------------------------------------------------------


class TestLoadIdxmlBuildsPsmsDf:
    def test_two_rows(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, proteins_df, sp_df, pg_df = load_identifications(path)
        assert len(psms_df) == 2, f"Expected 2 rows, got {len(psms_df)}"

    def test_peptidoform_unimod(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        # hit0 — S(Phospho) → S[UNIMOD:21]
        pf0 = psms_df.loc[psms_df["hit_index"] == 0, "peptidoform"].iloc[0]
        assert "UNIMOD:21" in pf0, f"Expected UNIMOD:21 in peptidoform, got {pf0!r}"

    def test_precursor_charge(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        charges = psms_df.sort_values("hit_index")["precursor_charge"].tolist()
        assert charges[0] == 2 and charges[1] == 3

    def test_scan_extracted_from_spectrum_reference(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        assert (psms_df["scan"] == 5).all(), "scan must be 5 for all hits"

    def test_score(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        scores = psms_df.sort_values("hit_index")["score"].tolist()
        assert abs(scores[0] - 0.01) < 1e-9 and abs(scores[1] - 0.05) < 1e-9

    def test_ascore_in_psm_metavalues(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        # hit0 carries AScore_site_scores
        mv0 = psms_df.loc[psms_df["hit_index"] == 0, "psm_metavalues"].iloc[0]
        names = [m["name"] for m in mv0]
        assert "AScore_site_scores" in names, f"AScore_site_scores missing; keys: {names}"

    def test_is_decoy_flag(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        sorted_df = psms_df.sort_values("hit_index")
        assert sorted_df["is_decoy"].tolist() == [False, True]

    def test_dtype_int32_columns(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        for col in ("scan", "precursor_charge", "hit_index", "peptide_identification_index"):
            assert psms_df[col].dtype == np.int32, f"{col} must be int32, got {psms_df[col].dtype}"

    def test_empty_dfs_returned(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        _, _, sp_df, pg_df = load_identifications(path)
        assert isinstance(sp_df, __import__("pandas").DataFrame)
        assert isinstance(pg_df, __import__("pandas").DataFrame)


# ---------------------------------------------------------------------------
# load_identifications — mzid
# ---------------------------------------------------------------------------


class TestLoadMzidBuildsPsmsDf:
    """Same assertions as idXML but stored/loaded via MzIdentMLFile."""

    def test_two_rows(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.mzid")
        oms.MzIdentMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        assert len(psms_df) == 2, f"Expected 2 rows, got {len(psms_df)}"

    def test_scan_extracted(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.mzid")
        oms.MzIdentMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        assert (psms_df["scan"] == 5).all(), "scan must be 5 for all hits"

    def test_peptidoform_unimod(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.mzid")
        oms.MzIdentMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        pf0 = psms_df.loc[psms_df["hit_index"] == 0, "peptidoform"].iloc[0]
        assert "UNIMOD:21" in pf0, f"Expected UNIMOD:21 in peptidoform, got {pf0!r}"

    def test_precursor_charge(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.mzid")
        oms.MzIdentMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        charges = psms_df.sort_values("hit_index")["precursor_charge"].tolist()
        assert charges[0] == 2 and charges[1] == 3

    def test_score(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.mzid")
        oms.MzIdentMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        scores = psms_df.sort_values("hit_index")["score"].tolist()
        assert abs(scores[0] - 0.01) < 1e-9 and abs(scores[1] - 0.05) < 1e-9

    def test_ascore_in_psm_metavalues(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list()
        path = str(tmp_path / "test.mzid")
        oms.MzIdentMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        mv0 = psms_df.loc[psms_df["hit_index"] == 0, "psm_metavalues"].iloc[0]
        names = [m["name"] for m in mv0]
        assert "AScore_site_scores" in names, f"AScore_site_scores missing; keys: {names}"
