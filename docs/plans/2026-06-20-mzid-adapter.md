# Format-agnostic idXML/mzIdentML I/O — Implementation Plan (revised)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `onsite all` (and each command) read/write idXML or mzIdentML by extension, computing AScore+PhosphoRS+LucXor and writing a scored mzid, with a flag- and data-gated Alanine-decoy mode.

**Architecture:** Format-agnostic `load_identifications` / `store_identifications` helpers (extension dispatch) wired into the three CLIs and `run_all_localizers`. No temp-file round-trip; no separate `mzid` command.

**Tech Stack:** Python 3.11, pyOpenMS 3.5.0 (`MzIdentMLFile`, `IdXMLFile`, `PeptideIdentificationList`), click, pytest.

## Global Constraints

- No new scoring logic — reuse `run_all_localizers` / the three CLIs.
- Score MetaValue names: `AScore_site_scores`, `PhosphoRS_site_probs`, `PhosphoRS_site_delta`, `Luciphor_site_scores`.
- pyOpenMS container facts (verified): `pep` must be a `PeptideIdentificationList` built via `push_back` (no `append`, no list constructor); `prot` is a plain Python list. `MzIdentMLFile`/`IdXMLFile` `load` and `store` both require these types.
- mzid store quirk (verified): `MzIdentMLFile().store()` raises `Invalid CV identifier!` if any `ProteinIdentification`'s `SearchParameters.variable_modifications` contains a non-UNIMOD custom modification (`PhosphoDecoy (A)` → `UNIMOD:99913`). Strip `PhosphoDecoy` entries from `variable_modifications` before an mzid store. The `PhosphoDecoy` modification ON the hits serializes fine and round-trips intact.
- Decoy gate: default no-decoy; `--add-decoys` + Ala-present = decoy pack; `--add-decoys` + no Ala = warn and fall back (never fail).
- Tests needing `data/1.mzML` must `pytest.skip` when absent.

## Status

- **Task 1 (DONE):** `run_all_localizers(...)` extracted from `all`; merge copies all four score keys. Commits `278289d..1e5201a`.
- **Task 2 (SUPERSEDED):** the committed `mzid_to_idxml`/`idxml_to_mzid` conversion helpers (commit `ae57e89`) are replaced by Task 2R below.

---

### Task 2R: Format-agnostic load/store helpers (replace conversion helpers)

**Files:**
- Modify: `onsite/mzid_adapter.py` (remove `mzid_to_idxml`, `idxml_to_mzid`, `LoadInfo`; add the helpers below)
- Modify: `tests/test_mzid_adapter.py` (remove the two conversion-roundtrip tests; add the tests below; keep `_has_score` and `_make_small_mzid` helpers — `_make_small_mzid` already strips PhosphoDecoy before its mzid store)

**Interfaces produced (later tasks depend on these):**
- `is_mzid(path: str) -> bool`
- `load_identifications(path: str) -> tuple[list, "PeptideIdentificationList"]`
- `store_identifications(path: str, prot: list, pep) -> None`
- `has_alanine(pep) -> bool`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mzid_adapter.py  (add; remove old roundtrip tests)
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py -k "roundtrip_idxml or phosphodecoy or has_alanine" -v`
Expected: FAIL (`ImportError` for the new names).

- [ ] **Step 3: Implement** (replace the conversion helpers in `onsite/mzid_adapter.py`)

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mzid_adapter.py -v`
Expected: all PASS (new tests pass; old conversion tests removed).

- [ ] **Step 5: Commit**

```bash
git add onsite/mzid_adapter.py tests/test_mzid_adapter.py
git commit -m "feat(mzid): format-agnostic load/store helpers (replace conversion helpers)"
```

---

### Task 3: Spectrum-reference validation

**Files:**
- Modify: `onsite/mzid_adapter.py`
- Test: `tests/test_mzid_adapter.py`

**Interfaces produced:**
- `SpectrumRefError(RuntimeError)`
- `validate_spectrum_refs(pep, mzml_path: str, min_match_fraction: float = 0.5) -> "ValidationResult"` with `ValidationResult(n_total, n_resolved, ok)`.

- [ ] **Step 1: Write failing test**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py::test_validate_spectrum_refs_raises_on_mismatch -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement** (append to `onsite/mzid_adapter.py`)

```python
from dataclasses import dataclass
from pyopenms import MSExperiment, FileHandler


class SpectrumRefError(RuntimeError):
    pass


@dataclass
class ValidationResult:
    n_total: int
    n_resolved: int
    ok: bool


def validate_spectrum_refs(pep, mzml_path: str, min_match_fraction: float = 0.5) -> ValidationResult:
    """Confirm PSM spectrum_reference values resolve to mzML spectrum native IDs.
    Raise SpectrumRefError when references exist but mostly fail to resolve."""
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_path, exp)
    native_ids = {s.getNativeID() for s in exp.getSpectra()}
    refs = [pid.getMetaValue("spectrum_reference")
            for pid in pep if pid.metaValueExists("spectrum_reference")]
    n_total = len(refs)
    n_resolved = sum(1 for r in refs if r in native_ids)
    ok = (n_total == 0) or (n_resolved / n_total >= min_match_fraction)
    if n_total > 0 and not ok:
        raise SpectrumRefError(
            f"Only {n_resolved}/{n_total} identification spectrum references resolve to "
            f"spectra in {mzml_path}. The identification file likely references spectra "
            f"by a scheme that does not match the mzML native IDs. Aborting to avoid "
            f"scoring zero PSMs."
        )
    return ValidationResult(n_total=n_total, n_resolved=n_resolved, ok=ok)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mzid_adapter.py::test_validate_spectrum_refs_raises_on_mismatch -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add onsite/mzid_adapter.py tests/test_mzid_adapter.py
git commit -m "feat(mzid): spectrum-reference validation against mzML native IDs"
```

---

### Task 4: Wire format-agnostic I/O into the three CLIs

**Files:**
- Modify: `onsite/ascore/cli.py`, `onsite/phosphors/cli.py`, `onsite/lucxor/cli.py`
- Test: `tests/test_mzid_adapter.py`

**Interfaces consumed:** `load_identifications`, `store_identifications` (Task 2R).

**Approach:** In each CLI, find where the identification input file is loaded
(currently `IdXMLFile().load(id_file, prot, pep)` or equivalent) and replace with
`prot, pep = load_identifications(id_file)`; find where the output is written
(currently `IdXMLFile().store(out, prot, pep)`) and replace with
`store_identifications(out, prot, pep)`. Import from `onsite.mzid_adapter`. Do not
change scoring, option definitions, or control flow — only the load/store calls.
The `--id-file`/`--out-file` (and lucxor's `--input-id`/`--output`) `click.Path`
types already accept any path; no option changes needed.

- [ ] **Step 1: Write failing test** (each CLI accepts mzid input and writes mzid output)

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py::test_ascore_accepts_mzid_in_and_out -v`
Expected: FAIL (ascore still hardcodes IdXMLFile → either errors on the mzid input or writes idXML content to the .mzid path so the mzid re-load fails) — or SKIP without mzML.

- [ ] **Step 3: Implement** — in each of the three CLIs:

For `onsite/ascore/cli.py` and `onsite/phosphors/cli.py`: locate `load_identifications`-equivalent (search for `IdXMLFile().load(` and `IdXMLFile().store(`) and replace:

```python
from onsite.mzid_adapter import load_identifications, store_identifications
# input load:
prot_ids, peptide_ids = load_identifications(id_file)
# output store:
store_identifications(out_file, processed_protein_ids, processed_peptide_ids)
```

(Use the existing local variable names in each file; the example names above are illustrative — match what the file already uses for the protein/peptide containers and the output path.)

For `onsite/lucxor/cli.py`: the input option is `input_id` and output is `output`; replace its `IdXMLFile().load(input_id, …)` and `IdXMLFile().store(output, …)` calls the same way.

- [ ] **Step 4: Run to verify pass + no regression**

Run: `python -m pytest tests/test_mzid_adapter.py::test_ascore_accepts_mzid_in_and_out tests/test_ascore.py tests/test_phosphors.py tests/test_lucxor.py -v`
Expected: new test PASSES (or SKIPs without mzML); per-algorithm suites still PASS.

- [ ] **Step 5: Commit**

```bash
git add onsite/ascore/cli.py onsite/phosphors/cli.py onsite/lucxor/cli.py tests/test_mzid_adapter.py
git commit -m "feat(mzid): wire format-agnostic load/store into the three CLIs"
```

---

### Task 5: `run_all_localizers` — decoy guard, spectrum validation, format-agnostic output

**Files:**
- Modify: `onsite/onsitec.py` (`run_all_localizers`, `merge_algorithm_results`)
- Test: `tests/test_mzid_adapter.py`

**Interfaces consumed:** `load_identifications`, `store_identifications`, `has_alanine`, `validate_spectrum_refs` (Tasks 2R-3).

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.skipif(not MZML.exists(), reason="data/1.mzML not present")
def test_all_id_mzid_out_mzid_has_three_scores(tmp_path):
    from onsite.mzid_adapter import load_identifications, store_identifications
    from onsite.onsitec import run_all_localizers
    from pyopenms import PeptideIdentificationList
    prot, pep = load_identifications(str(IDXML))
    small = PeptideIdentificationList()
    for i in range(min(5, pep.size())):
        small.push_back(pep.at(i))
    in_mzid = str(tmp_path / "in.mzid"); store_identifications(in_mzid, prot, small)
    out_mzid = str(tmp_path / "scored.mzid")
    run_all_localizers(str(MZML), in_mzid, out_mzid, threads=1)
    _, outpep = load_identifications(out_mzid)
    assert _has_score(outpep, "AScore_site_scores")
    assert _has_score(outpep, "PhosphoRS_site_probs")
    assert _has_score(outpep, "Luciphor_site_scores")


@pytest.mark.skipif(not MZML.exists(), reason="data/1.mzML not present")
def test_add_decoys_falls_back_when_no_ala(tmp_path, monkeypatch):
    from onsite import mzid_adapter
    from onsite.onsitec import run_all_localizers
    monkeypatch.setattr(mzid_adapter, "has_alanine", lambda pep: False)
    # also patch the name imported into onsitec if imported directly:
    import onsite.onsitec as oc
    if hasattr(oc, "has_alanine"):
        monkeypatch.setattr(oc, "has_alanine", lambda pep: False)
    prot, pep = mzid_adapter.load_identifications(str(IDXML))
    from pyopenms import PeptideIdentificationList
    small = PeptideIdentificationList()
    for i in range(min(3, pep.size())):
        small.push_back(pep.at(i))
    in_idxml = str(tmp_path / "in.idXML"); mzid_adapter.store_identifications(in_idxml, prot, small)
    out_idxml = str(tmp_path / "out.idXML")
    # should not raise and should complete in no-decoy mode
    run_all_localizers(str(MZML), in_idxml, out_idxml, threads=1, add_decoys=True)
    assert os.path.exists(out_idxml)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py -k "id_mzid_out_mzid or falls_back" -v`
Expected: the mzid-out test FAILS because `merge_algorithm_results` writes via `IdXMLFile` (the `.mzid` output is idXML content → re-load as mzid fails) — or SKIP without mzML.

- [ ] **Step 3: Implement** in `onsite/onsitec.py`:

At the top of `run_all_localizers`, before the tempdir block, add the guard and validation:

```python
from onsite.mzid_adapter import (
    load_identifications, store_identifications, has_alanine, validate_spectrum_refs,
)

# Load identifications once for the decoy guard + spectrum-reference validation.
_prot, _pep = load_identifications(id_file)
validate_spectrum_refs(_pep, in_file)
if add_decoys and not has_alanine(_pep):
    click.echo("Warning: --add-decoys set but no Alanine candidate present; "
               "computing the no-decoy score pack instead.")
    add_decoys = False
```

In `merge_algorithm_results`, replace the final `IdXMLFile().store(output_file, merged_prot_ids, merged_pep_ids)` (match the file's actual variable names) with:

```python
from onsite.mzid_adapter import store_identifications
store_identifications(output_file, merged_prot_ids, merged_pep_ids)
```

(The three intermediate per-tool files written inside the tempdir stay `.idXML`
and keep using `IdXMLFile` — they are internal. The CLIs themselves now use
`store_identifications` from Task 4, but since the intermediate paths end in
`.idXML`, that writes idXML as before.)

- [ ] **Step 4: Run to verify pass + full suite**

Run: `python -m pytest tests/test_mzid_adapter.py tests/test_cli.py -v`
Then: `python -m pytest tests/ -q`
Expected: all PASS (mzML-dependent tests may SKIP).

- [ ] **Step 5: Commit**

```bash
git add onsite/onsitec.py tests/test_mzid_adapter.py
git commit -m "feat(mzid): decoy guard + spectrum validation + format-agnostic output in run_all_localizers"
```

---

## Self-Review

- **Coverage:** helpers (Task 2R), spectrum validation (Task 3), CLI wiring for both I/O directions (Task 4), decoy guard + output format + validation in the pipeline (Task 5). Decoy gate, mzid-store quirk, container types — all in Global Constraints and exercised by tests.
- **Placeholders:** none. The Task 4 Step-1 test has an inline note to remove a deliberately-flagged stray import line; the corrected import is stated.
- **Type consistency:** `load_identifications -> (list, PeptideIdentificationList)`, `store_identifications(path, prot, pep)`, `has_alanine(pep) -> bool`, `validate_spectrum_refs(pep, mzml, ...) -> ValidationResult`, `SpectrumRefError` — used consistently across Tasks 2R-5.
