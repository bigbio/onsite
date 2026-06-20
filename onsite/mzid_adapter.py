"""Format-agnostic identification I/O for the onsite localizer pipeline."""
import os
from pyopenms import IdXMLFile, MzIdentMLFile, PeptideIdentificationList

_MZID_EXTS = (".mzid", ".mzidentml")


def is_mzid(path: str) -> bool:
    p = path.lower()
    return p.endswith(_MZID_EXTS)


def load_identifications(path: str):
    """Load idXML or mzIdentML by extension. Returns (prot_list, PeptideIdentificationList)."""
    prot = []
    pep = PeptideIdentificationList()
    if is_mzid(path):
        MzIdentMLFile().load(path, prot, pep)
    else:
        IdXMLFile().load(path, prot, pep)
    return prot, pep


def _strip_custom_variable_mods(prot):
    """Remove non-UNIMOD custom modifications (PhosphoDecoy) from each protein's
    SearchParameters.variable_modifications so MzIdentMLFile().store() does not
    reject them ('Invalid CV identifier!'). The modification on the hits is kept."""
    for p in prot:
        sp = p.getSearchParameters()
        kept = [m for m in sp.variable_modifications
                if b"PhosphoDecoy" not in (m if isinstance(m, bytes) else m.encode())]
        sp.variable_modifications = kept
        p.setSearchParameters(sp)


def store_identifications(path: str, prot, pep) -> None:
    """Store idXML or mzIdentML by extension. For mzid, strip custom variable mods first."""
    if is_mzid(path):
        _strip_custom_variable_mods(prot)
        MzIdentMLFile().store(path, prot, pep)
    else:
        IdXMLFile().store(path, prot, pep)


def has_alanine(pep) -> bool:
    """True if any peptide hit's unmodified sequence contains an 'A' residue."""
    for pid in pep:
        for hit in pid.getHits():
            if "A" in hit.getSequence().toUnmodifiedString():
                return True
    return False
