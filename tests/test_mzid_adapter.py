import os
import pytest
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
MZML = DATA / "1.mzML"
IDXML = DATA / "1_consensus_fdr_filter_pep.idXML"


def _make_small_mzid(tmp_path, n=5):
    """Build a small mzid fixture by trimming the committed idXML and round-tripping."""
    from pyopenms import IdXMLFile, MzIdentMLFile, PeptideIdentificationList
    prot, pep = [], PeptideIdentificationList()
    IdXMLFile().load(str(IDXML), prot, pep)
    # Strip custom PhosphoDecoy modification so MzIdentMLFile.store() succeeds
    if prot:
        sp = prot[0].getSearchParameters()
        sp.variable_modifications = [
            m for m in sp.variable_modifications if b"PhosphoDecoy" not in m
        ]
        prot[0].setSearchParameters(sp)
    small = PeptideIdentificationList()
    for i in range(n):
        small.push_back(pep[i])
    small_idxml = str(tmp_path / "small.idXML")
    IdXMLFile().store(small_idxml, prot, small)
    mzid = str(tmp_path / "small.mzid")
    MzIdentMLFile().store(mzid, prot, small)
    return small_idxml, mzid, n


def _has_score(pep_ids, meta):
    for pid in pep_ids:
        for hit in pid.getHits():
            if hit.metaValueExists(meta):
                return True
    return False


def test_load_store_roundtrip_idxml(tmp_path):
    from onsite.mzid_adapter import load_identifications, store_identifications
    prot, pep = load_identifications(str(IDXML))
    n = sum(len(p.getHits()) for p in pep)
    out = str(tmp_path / "rt.idXML")
    store_identifications(out, prot, pep)
    prot2, pep2 = load_identifications(out)
    assert sum(len(p.getHits()) for p in pep2) == n


def test_store_mzid_with_phosphodecoy_roundtrips(tmp_path):
    from onsite.mzid_adapter import store_identifications, load_identifications
    from pyopenms import (AASequence, PeptideHit, PeptideIdentification,
                          ProteinIdentification, PeptideIdentificationList)
    hit = PeptideHit(); hit.setSequence(AASequence.fromString("AS(Phospho)A(PhosphoDecoy)K"))
    hit.setScore(1.0); hit.setCharge(2); hit.setMetaValue("AScore_site_scores", "{1: 50.0}")
    pid = PeptideIdentification(); pid.setHits([hit]); pid.setScoreType("AScore")
    pid.setMetaValue("spectrum_reference", "scan=1")
    prot = ProteinIdentification(); sp = prot.getSearchParameters()
    sp.variable_modifications = [b"Phospho (S)", b"PhosphoDecoy (A)"]; prot.setSearchParameters(sp)
    pep = PeptideIdentificationList(); pep.push_back(pid)
    out = str(tmp_path / "decoy.mzid")
    store_identifications(out, [prot], pep)  # must not raise
    prot2, pep2 = load_identifications(out)
    h = pep2.at(0).getHits()[0]
    assert "PhosphoDecoy" in h.getSequence().toString()
    assert h.getMetaValue("AScore_site_scores") == "{1: 50.0}"


def test_has_alanine(tmp_path):
    from onsite.mzid_adapter import has_alanine
    from pyopenms import AASequence, PeptideHit, PeptideIdentification, PeptideIdentificationList
    def mk(seq):
        hit = PeptideHit(); hit.setSequence(AASequence.fromString(seq))
        pid = PeptideIdentification(); pid.setHits([hit])
        pep = PeptideIdentificationList(); pep.push_back(pid); return pep
    assert has_alanine(mk("AS(Phospho)K")) is True
    assert has_alanine(mk("S(Phospho)PEK")) is False


def test_validate_spectrum_refs_raises_on_mismatch(tmp_path):
    from onsite.mzid_adapter import validate_spectrum_refs, SpectrumRefError, load_identifications
    from pyopenms import MSExperiment, FileHandler
    prot, pep = load_identifications(str(IDXML))
    # keep one PSM, point it at a non-existent spectrum
    one = pep.at(0); one.setMetaValue("spectrum_reference", "scan=does-not-exist-999999")
    from pyopenms import PeptideIdentificationList
    pep1 = PeptideIdentificationList(); pep1.push_back(one)
    empty_mzml = str(tmp_path / "empty.mzML")
    FileHandler().storeExperiment(empty_mzml, MSExperiment())
    with pytest.raises(SpectrumRefError):
        validate_spectrum_refs(pep1, empty_mzml)


@pytest.mark.skipif(not MZML.exists(), reason="data/1.mzML not present")
def test_ascore_accepts_mzid_in_and_out(tmp_path):
    from onsite.mzid_adapter import load_identifications, store_identifications
    from pyopenms import PeptideIdentificationList
    from click.testing import CliRunner
    from onsite.ascore.cli import ascore
    # build a small mzid input from a few PSMs of the committed idXML
    prot, pep = load_identifications(str(IDXML))
    small = PeptideIdentificationList()
    for i in range(min(5, pep.size())):
        small.push_back(pep.at(i))
    in_mzid = str(tmp_path / "in.mzid"); store_identifications(in_mzid, prot, small)
    out_mzid = str(tmp_path / "out.mzid")
    res = CliRunner().invoke(ascore, ["-in", str(MZML), "-id", in_mzid,
                                      "-out", out_mzid, "--threads", "1"],
                             catch_exceptions=False)
    assert res.exit_code == 0
    assert os.path.exists(out_mzid)
    _, outpep = load_identifications(out_mzid)
    assert _has_score(outpep, "AScore_site_scores")


@pytest.mark.skipif(not MZML.exists(), reason="data/1.mzML not present")
def test_run_all_localizers_writes_three_scores(tmp_path):
    from onsite.onsitec import run_all_localizers
    from pyopenms import IdXMLFile

    out = str(tmp_path / "merged.idXML")
    run_all_localizers(str(MZML), str(IDXML), out, threads=1)

    from pyopenms import PeptideIdentificationList
    prot, pep = [], PeptideIdentificationList()
    IdXMLFile().load(out, prot, pep)
    assert _has_score(pep, "AScore_site_scores")
    assert _has_score(pep, "PhosphoRS_site_probs")
    assert _has_score(pep, "Luciphor_site_scores")
