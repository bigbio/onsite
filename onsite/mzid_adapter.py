"""Format-agnostic identification I/O for the onsite localizer pipeline."""
import re
from dataclasses import dataclass
from pyopenms import IdXMLFile, MzIdentMLFile, MSExperiment, FileHandler, PeptideIdentificationList

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
    reject them ('Invalid CV identifier!'). The modification on the hits is kept.
    Note: mutates the passed-in ``prot`` objects in place."""
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


_SCAN_RE = re.compile(r"scan=(\d+)")


def _extract_scan_number(ref: str):
    """Extract the integer scan number from a spectrum reference string.

    Handles both compact forms (``"scan=4886"``) and full mzML nativeID strings
    (``"controllerType=0 controllerNumber=1 scan=4886"``).  Returns ``None``
    when no ``scan=<int>`` token is present.
    """
    m = _SCAN_RE.search(ref)
    return int(m.group(1)) if m else None


class SpectrumRefError(RuntimeError):
    pass


@dataclass
class ValidationResult:
    n_total: int
    n_resolved: int
    ok: bool


def validate_spectrum_refs(pep, mzml_path: str, min_match_fraction: float = 0.5) -> ValidationResult:
    """Confirm PSM spectrum_reference values resolve to mzML spectrum native IDs.

    Matching is scan-number-tolerant: a reference is considered resolved when
    EITHER (a) it matches a native ID by exact string, OR (b) the integer scan
    number embedded in the reference (``scan=N``) matches the scan number
    embedded in any native ID.  This mirrors the behaviour of
    ``extract_scan_number_from_reference`` / ``find_spectrum_by_scan`` in the
    phosphors resolver, so a compact ref ``"scan=4886"`` correctly resolves
    against a full mzML nativeID
    ``"controllerType=0 controllerNumber=1 scan=4886"``.

    Raises SpectrumRefError when references exist but mostly fail to resolve.
    """
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_path, exp)
    native_ids = {s.getNativeID() for s in exp.getSpectra()}
    # Build a set of integer scan numbers from native IDs once for efficiency.
    native_scan_numbers = {
        _extract_scan_number(nid)
        for nid in native_ids
        if _extract_scan_number(nid) is not None
    }
    refs = [pid.getMetaValue("spectrum_reference")
            for pid in pep if pid.metaValueExists("spectrum_reference")]
    n_total = len(refs)

    def _resolves(ref: str) -> bool:
        # (a) exact string match
        if ref in native_ids:
            return True
        # (b) scan-number match
        scan = _extract_scan_number(ref)
        return scan is not None and scan in native_scan_numbers

    n_resolved = sum(1 for r in refs if _resolves(r))
    ok = (n_total == 0) or (n_resolved / n_total >= min_match_fraction)
    if n_total > 0 and not ok:
        raise SpectrumRefError(
            f"Only {n_resolved}/{n_total} identification spectrum references resolve to "
            f"spectra in {mzml_path}. The identification file likely references spectra "
            f"by a scheme that does not match the mzML native IDs. Aborting to avoid "
            f"scoring zero PSMs."
        )
    return ValidationResult(n_total=n_total, n_resolved=n_resolved, ok=ok)
