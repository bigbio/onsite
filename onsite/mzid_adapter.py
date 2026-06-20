"""Adapter bridging mzIdentML to the idXML-based onsite localizer pipeline."""
from dataclasses import dataclass
from pyopenms import IdXMLFile, MzIdentMLFile, PeptideIdentificationList

TARGET_RESIDUES = set("STY")


@dataclass
class LoadInfo:
    n_psms: int
    has_ala: bool


def _has_alanine(pep_ids) -> bool:
    """True if any peptide hit's unmodified sequence contains an 'A' residue
    (the practical precondition for Ala-decoy scoring to do anything)."""
    for pid in pep_ids:
        for hit in pid.getHits():
            if "A" in hit.getSequence().toUnmodifiedString():
                return True
    return False


def mzid_to_idxml(mzid_path: str, idxml_path: str) -> LoadInfo:
    prot, pep = [], PeptideIdentificationList()
    MzIdentMLFile().load(mzid_path, prot, pep)
    IdXMLFile().store(idxml_path, prot, pep)
    n = sum(len(pid.getHits()) for pid in pep)
    return LoadInfo(n_psms=n, has_ala=_has_alanine(pep))


def idxml_to_mzid(idxml_path: str, mzid_path: str) -> None:
    prot, pep = [], PeptideIdentificationList()
    IdXMLFile().load(idxml_path, prot, pep)
    MzIdentMLFile().store(mzid_path, prot, pep)
