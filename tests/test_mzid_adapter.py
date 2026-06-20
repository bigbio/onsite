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


def test_mzid_to_idxml_roundtrip_preserves_psms(tmp_path):
    from onsite.mzid_adapter import mzid_to_idxml
    from pyopenms import IdXMLFile, PeptideIdentificationList
    _, mzid, n = _make_small_mzid(tmp_path)
    out_idxml = str(tmp_path / "back.idXML")
    info = mzid_to_idxml(mzid, out_idxml)
    assert info.n_psms == n
    prot, pep = [], PeptideIdentificationList()
    IdXMLFile().load(out_idxml, prot, pep)
    assert len(pep) == n


def test_idxml_to_mzid_creates_file(tmp_path):
    from onsite.mzid_adapter import idxml_to_mzid
    small_idxml, _, _ = _make_small_mzid(tmp_path)
    out_mzid = str(tmp_path / "out.mzid")
    idxml_to_mzid(small_idxml, out_mzid)
    assert os.path.exists(out_mzid) and os.path.getsize(out_mzid) > 0


def _has_score(pep_ids, meta):
    for pid in pep_ids:
        for hit in pid.getHits():
            if hit.metaValueExists(meta):
                return True
    return False


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
