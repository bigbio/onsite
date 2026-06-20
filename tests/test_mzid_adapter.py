import os
import pytest
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
MZML = DATA / "1.mzML"
IDXML = DATA / "1_consensus_fdr_filter_pep.idXML"


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
