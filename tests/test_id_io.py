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


# ---------------------------------------------------------------------------
# Helper — build pep list with a known protein accession
# ---------------------------------------------------------------------------


def _build_pep_list_with_protein():
    """Return (prot_list, PeptideIdentificationList) with 1 hit linked to PROT_A."""
    import pyopenms as oms

    pi = oms.ProteinIdentification()
    pi.setIdentifier("run1")
    pi.setScoreType("Percolator_qvalue")
    pi.setHigherScoreBetter(False)
    ph = oms.ProteinHit()
    ph.setAccession("PROT_A")
    pi.setHits([ph])
    prot = [pi]

    pep_list = oms.PeptideIdentificationList()
    pid = oms.PeptideIdentification()
    pid.setScoreType("Percolator_qvalue")
    pid.setHigherScoreBetter(False)
    pid.setRT(100.5)
    pid.setMZ(500.0)
    pid.setIdentifier("run1")
    pid.setMetaValue("spectrum_reference", "controllerType=0 scan=7")

    hit = oms.PeptideHit()
    hit.setSequence(oms.AASequence.fromString("S(Phospho)PEK"))
    hit.setCharge(2)
    hit.setScore(0.01)
    hit.setMetaValue("target_decoy", "target")

    ev = oms.PeptideEvidence()
    ev.setProteinAccession("PROT_A")
    ev.setAABefore(b"K")
    ev.setAAAfter(b"L")
    ev.setStart(10)
    ev.setEnd(13)
    hit.setPeptideEvidences([ev])

    pid.setHits([hit])
    pep_list.push_back(pid)
    return prot, pep_list


def _build_pep_list_with_numeric_metavalues():
    """Return (prot_list, PeptideIdentificationList) with 1 hit that has numeric metavalues."""
    import pyopenms as oms

    pi = oms.ProteinIdentification()
    pi.setIdentifier("run1")
    pi.setScoreType("Percolator_qvalue")
    pi.setHigherScoreBetter(False)
    prot = [pi]

    pep_list = oms.PeptideIdentificationList()
    pid = oms.PeptideIdentification()
    pid.setScoreType("Percolator_qvalue")
    pid.setHigherScoreBetter(False)
    pid.setRT(200.0)
    pid.setMZ(600.0)
    pid.setIdentifier("run1")
    pid.setMetaValue("spectrum_reference", "scan=3")

    hit = oms.PeptideHit()
    hit.setSequence(oms.AASequence.fromString("PEK"))
    hit.setCharge(2)
    hit.setScore(0.02)
    hit.setMetaValue("target_decoy", "target")
    hit.setMetaValue("xcorr_score", 4.567)   # float metavalue
    hit.setMetaValue("num_matched_ions", 6)    # int metavalue

    pid.setHits([hit])
    pep_list.push_back(pid)
    return prot, pep_list


# ---------------------------------------------------------------------------
# New assertions: protein_accessions, metavalue value_type, modifications,
# detect_format(Path)
# ---------------------------------------------------------------------------


class TestDetectFormatPathlib:
    """detect_format must accept pathlib.Path objects."""

    def test_directory_pathlib(self, tmp_path):
        d = tmp_path / "foo"
        d.mkdir()
        assert detect_format(d) == "idparquet"

    def test_idparquet_extension_pathlib(self, tmp_path):
        p = tmp_path / "foo.idparquet"
        assert detect_format(p) == "idparquet"

    def test_idxml_pathlib(self, tmp_path):
        p = tmp_path / "foo.idXML"
        assert detect_format(p) == "idxml"

    def test_mzid_pathlib(self, tmp_path):
        p = tmp_path / "foo.mzid"
        assert detect_format(p) == "mzid"


class TestProteinAccessions:
    """protein_accessions must be populated from PeptideEvidences."""

    def test_protein_accessions_populated(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_protein()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        pa = psms_df["protein_accessions"].iloc[0]
        assert isinstance(pa, np.ndarray), f"Expected ndarray, got {type(pa)}"
        assert len(pa) >= 1, "Expected at least one protein accession entry"
        accessions = [d["accession"] for d in pa]
        assert "PROT_A" in accessions, f"PROT_A not in {accessions}"

    def test_protein_accessions_entry_has_required_keys(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_protein()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        pa = psms_df["protein_accessions"].iloc[0]
        entry = pa[0]
        for key in ("accession", "aa_before", "aa_after", "start", "end"):
            assert key in entry, f"Missing key {key!r} in protein_accessions entry"


class TestMetavalueType:
    """Numeric metavalues must have value_type 'double' or 'int', not 'string'."""

    def test_float_metavalue_has_double_type(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_numeric_metavalues()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        mv = psms_df["psm_metavalues"].iloc[0]
        by_name = {m["name"]: m for m in mv}
        assert "xcorr_score" in by_name, f"xcorr_score not in metavalues: {list(by_name)}"
        assert by_name["xcorr_score"]["value_type"] == "double", (
            f"Expected 'double', got {by_name['xcorr_score']['value_type']!r}"
        )

    def test_int_metavalue_has_int_type(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_numeric_metavalues()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        mv = psms_df["psm_metavalues"].iloc[0]
        by_name = {m["name"]: m for m in mv}
        assert "num_matched_ions" in by_name, f"num_matched_ions not in metavalues: {list(by_name)}"
        assert by_name["num_matched_ions"]["value_type"] == "int", (
            f"Expected 'int', got {by_name['num_matched_ions']['value_type']!r}"
        )

    def test_string_metavalue_remains_string(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_numeric_metavalues()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        mv = psms_df["psm_metavalues"].iloc[0]
        by_name = {m["name"]: m for m in mv}
        assert "target_decoy" in by_name, f"target_decoy not in metavalues: {list(by_name)}"
        assert by_name["target_decoy"]["value_type"] == "string", (
            f"Expected 'string', got {by_name['target_decoy']['value_type']!r}"
        )


class TestModificationsPopulated:
    """modifications must be non-empty for a modified peptidoform."""

    def test_phospho_modifications_non_empty(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_protein()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        # hit 0 has S(Phospho)PEK — modifications should be non-empty
        mods = psms_df["modifications"].iloc[0]
        assert isinstance(mods, np.ndarray), f"Expected ndarray, got {type(mods)}"
        assert len(mods) > 0, "modifications must be non-empty for S(Phospho)PEK"

    def test_unmodified_peptide_modifications_empty(self, tmp_path):
        import pyopenms as oms

        prot, pep_list = _build_pep_list_with_numeric_metavalues()
        path = str(tmp_path / "test.idXML")
        oms.IdXMLFile().store(path, prot, pep_list)

        psms_df, *_ = load_identifications(path)
        # PEK has no modifications
        mods = psms_df["modifications"].iloc[0]
        assert isinstance(mods, np.ndarray), f"Expected ndarray, got {type(mods)}"
        assert len(mods) == 0, f"Unmodified PEK should have empty modifications, got {mods}"
