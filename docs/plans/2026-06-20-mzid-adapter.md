# mzIdentML Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `onsite mzid` command that computes AScore + PhosphoRS + LucXor scores for an mzIdentML+mzML pair and writes a scored mzIdentML, with a flag- and data-gated Alanine-decoy mode.

**Architecture:** Convert-and-wrap. A thin adapter converts mzIdentML→idXML (pyOpenMS), runs the existing `onsite all` orchestration (refactored into a reusable `run_all_localizers` function), then converts the merged idXML→mzIdentML. Scores attached as `PeptideHit` MetaValues serialize to mzid `userParam`s automatically.

**Tech Stack:** Python 3.11, pyOpenMS 3.5.0 (`MzIdentMLFile`, `IdXMLFile`, `MSExperiment`/`FileHandler`), click, pytest.

## Global Constraints

- No new scoring logic — reuse `onsite all` / `run_all_localizers` verbatim.
- Output mzid is pyOpenMS-regenerated (carries IDs + score userParams; not byte-preserving).
- One decoy mode per output file: default = no-decoy pack; `--add-decoys` + Ala-present = decoy pack; `--add-decoys` + no Ala = warn and fall back to no-decoy pack (never fail for this reason).
- Score MetaValue names already produced by the pipeline: `AScore_site_scores`, `PhosphoRS_site_probs`, `PhosphoRS_site_delta`, `Luciphor_site_scores`.
- Follow existing CLI option conventions (`-in/--in-file`, `-id/--id-file`, `-out/--out-file`, `--fragment-mass-tolerance` default 0.05, `--fragment-mass-unit [Da|ppm]` default Da, `--threads` default 1, `--add-decoys`).
- Tests that need `data/1.mzML` must `pytest.skip` when it is absent (it is gitignored), mirroring `tests/test_algorithm_comparison.py`.

## File Structure

- Modify `onsite/onsitec.py` — extract `run_all_localizers(...)` from the `all` command body; register the new `mzid` command.
- Create `onsite/mzid_adapter.py` — format-bridge helpers + `run(...)` orchestration.
- Create `onsite/mzid/__init__.py` + `onsite/mzid/cli.py`? **No** — keep it a single module + a command in `onsitec.py` (YAGNI; mirrors how `all` lives in `onsitec.py`).
- Modify `pyproject.toml` — add `onsite-mzid` console script.
- Create `tests/test_mzid_adapter.py` — unit + end-to-end tests.

---

### Task 1: Extract `run_all_localizers` from the `all` command

**Files:**
- Modify: `onsite/onsitec.py:102-237` (the `all` command body)
- Test: `tests/test_mzid_adapter.py`

**Interfaces:**
- Produces: `run_all_localizers(in_file: str, id_file: str, out_file: str, fragment_mass_tolerance: float = 0.05, fragment_mass_unit: str = "Da", threads: int = 1, add_decoys: bool = False, debug: bool = False) -> None` — runs the three localizers into a tempdir and merges into `out_file` (an idXML). Same behavior as today's `all`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mzid_adapter.py
import os
import pytest
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
MZML = DATA / "1.mzML"
IDXML = DATA / "1_consensus_fdr_filter_pep.idXML"


def _has_score(pep_ids, meta):
    from pyopenms import IdXMLFile  # noqa
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

    prot, pep = [], []
    IdXMLFile().load(out, prot, pep)
    assert _has_score(pep, "AScore_site_scores")
    assert _has_score(pep, "PhosphoRS_site_probs")
    assert _has_score(pep, "Luciphor_site_scores")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mzid_adapter.py::test_run_all_localizers_writes_three_scores -v`
Expected: FAIL with `ImportError: cannot import name 'run_all_localizers'` (or SKIP if `data/1.mzML` absent — if skipped, extract it first: `cd data && cat 1.z01 1.z02 1.z03 1.zip > /tmp/c.zip && unzip -o /tmp/c.zip && rm /tmp/c.zip`).

- [ ] **Step 3: Extract the function**

In `onsite/onsitec.py`, add this function above the `all` command, moving the tempdir/invoke/merge body verbatim out of `all` (keep all `ctx.invoke(...)` argument blocks exactly as they are today, including the LucXor parameter list and the `add_decoys`→`target_mods` branch):

```python
def run_all_localizers(
    in_file,
    id_file,
    out_file,
    fragment_mass_tolerance=0.05,
    fragment_mass_unit="Da",
    threads=1,
    add_decoys=False,
    debug=False,
):
    """Run AScore + PhosphoRS + LucXor into a tempdir and merge into out_file (idXML).

    Extracted from the `all` command so other entry points (e.g. the mzid
    adapter) can reuse the exact same orchestration. Behavior is unchanged.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ascore_out = os.path.join(tmpdir, "ascore_result.idXML")
        phosphors_out = os.path.join(tmpdir, "phosphors_result.idXML")
        lucxor_out = os.path.join(tmpdir, "lucxor_result.idXML")

        from onsite.ascore.cli import ascore as ascore_func
        ctx = click.Context(ascore_func)
        ctx.invoke(
            ascore_func,
            in_file=in_file, id_file=id_file, out_file=ascore_out,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_mass_unit=fragment_mass_unit,
            threads=threads, debug=debug, add_decoys=add_decoys,
        )

        from onsite.phosphors.cli import phosphors as phosphors_func
        ctx = click.Context(phosphors_func)
        ctx.invoke(
            phosphors_func,
            in_file=in_file, id_file=id_file, out_file=phosphors_out,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_mass_unit=fragment_mass_unit,
            threads=threads, debug=debug, add_decoys=add_decoys,
        )

        from onsite.lucxor.cli import lucxor as lucxor_func
        if add_decoys:
            target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)", "PhosphoDecoy (A)")
        else:
            target_mods = ("Phospho (S)", "Phospho (T)", "Phospho (Y)")
        ctx = click.Context(lucxor_func)
        ctx.invoke(
            lucxor_func,
            input_spectrum=in_file, input_id=id_file, output=lucxor_out,
            fragment_method="CID",
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_error_units=fragment_mass_unit,
            min_mz=150.0, target_modifications=target_mods,
            neutral_losses=("sty -H3PO4 -97.97690",),
            decoy_mass=79.966331,
            decoy_neutral_losses=("X -H3PO4 -97.97690",),
            max_charge_state=5, max_peptide_length=40, max_num_perm=16384,
            modeling_score_threshold=0.95, scoring_threshold=0.0,
            min_num_psms_model=50, threads=threads, rt_tolerance=0.01,
            debug=debug, log_file=None, disable_split_by_charge=False,
        )

        merge_algorithm_results(ascore_out, phosphors_out, lucxor_out, out_file)
```

Then replace the body of the `all` command (between the `click.echo` banner and the `elapsed = ...` line) with a single call:

```python
        run_all_localizers(
            in_file, id_file, out_file,
            fragment_mass_tolerance, fragment_mass_unit, threads, add_decoys, debug,
        )
```

Keep the `try/except`, banners, and timing in `all` as-is.

- [ ] **Step 4: Run test + existing suite to verify pass and no regression**

Run: `python -m pytest tests/test_mzid_adapter.py::test_run_all_localizers_writes_three_scores tests/test_cli.py -v`
Expected: the new test PASSES (or SKIPs if no mzML); `test_cli.py` still PASSES.

- [ ] **Step 5: Commit**

```bash
git add onsite/onsitec.py tests/test_mzid_adapter.py
git commit -m "refactor(onsitec): extract run_all_localizers from all command"
```

---

### Task 2: mzid↔idXML conversion helpers

**Files:**
- Create: `onsite/mzid_adapter.py`
- Test: `tests/test_mzid_adapter.py`

**Interfaces:**
- Produces:
  - `mzid_to_idxml(mzid_path: str, idxml_path: str) -> "LoadInfo"` — loads mzid, stores idXML, returns `LoadInfo(n_psms: int, has_ala: bool)`.
  - `idxml_to_mzid(idxml_path: str, mzid_path: str) -> None` — loads idXML, stores mzid.
  - `LoadInfo` dataclass with fields `n_psms: int`, `has_ala: bool`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_mzid_adapter.py
def _make_small_mzid(tmp_path, n=5):
    """Build a small mzid fixture by trimming the committed idXML and round-tripping."""
    from pyopenms import IdXMLFile, MzIdentMLFile
    prot, pep = [], []
    IdXMLFile().load(str(IDXML), prot, pep)
    small = pep[:n]
    small_idxml = str(tmp_path / "small.idXML")
    IdXMLFile().store(small_idxml, prot, small)
    mzid = str(tmp_path / "small.mzid")
    MzIdentMLFile().store(mzid, prot, small)
    return small_idxml, mzid, len(small)


def test_mzid_to_idxml_roundtrip_preserves_psms(tmp_path):
    from onsite.mzid_adapter import mzid_to_idxml
    from pyopenms import IdXMLFile
    _, mzid, n = _make_small_mzid(tmp_path)
    out_idxml = str(tmp_path / "back.idXML")
    info = mzid_to_idxml(mzid, out_idxml)
    assert info.n_psms == n
    prot, pep = [], []
    IdXMLFile().load(out_idxml, prot, pep)
    assert len(pep) == n


def test_idxml_to_mzid_creates_file(tmp_path):
    from onsite.mzid_adapter import idxml_to_mzid
    small_idxml, _, _ = _make_small_mzid(tmp_path)
    out_mzid = str(tmp_path / "out.mzid")
    idxml_to_mzid(small_idxml, out_mzid)
    assert os.path.exists(out_mzid) and os.path.getsize(out_mzid) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py -k "roundtrip or creates_file" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'onsite.mzid_adapter'`.

- [ ] **Step 3: Implement the helpers**

```python
# onsite/mzid_adapter.py
"""Adapter bridging mzIdentML to the idXML-based onsite localizer pipeline."""
from dataclasses import dataclass
from pyopenms import IdXMLFile, MzIdentMLFile

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
    prot, pep = [], []
    MzIdentMLFile().load(mzid_path, prot, pep)
    IdXMLFile().store(idxml_path, prot, pep)
    n = sum(len(pid.getHits()) for pid in pep)
    return LoadInfo(n_psms=n, has_ala=_has_alanine(pep))


def idxml_to_mzid(idxml_path: str, mzid_path: str) -> None:
    prot, pep = [], []
    IdXMLFile().load(idxml_path, prot, pep)
    MzIdentMLFile().store(mzid_path, prot, pep)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mzid_adapter.py -k "roundtrip or creates_file" -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add onsite/mzid_adapter.py tests/test_mzid_adapter.py
git commit -m "feat(mzid): mzid<->idXML conversion helpers with Ala detection"
```

---

### Task 3: Spectrum-reference validation

**Files:**
- Modify: `onsite/mzid_adapter.py`
- Test: `tests/test_mzid_adapter.py`

**Interfaces:**
- Consumes: pyOpenMS `MSExperiment`, `FileHandler`.
- Produces: `validate_spectrum_refs(idxml_path: str, mzml_path: str, min_match_fraction: float = 0.5) -> "ValidationResult"` where `ValidationResult` has `n_total: int`, `n_resolved: int`, `ok: bool`. Raises `SpectrumRefError(message)` when `n_total > 0` and the resolved fraction `< min_match_fraction`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_mzid_adapter.py
def test_validate_spectrum_refs_raises_on_mismatch(tmp_path):
    from onsite.mzid_adapter import validate_spectrum_refs, SpectrumRefError
    from pyopenms import IdXMLFile, MSExperiment, MSSpectrum, FileHandler
    # idXML with one PSM whose spectrum_reference cannot exist in an empty mzML
    prot, pep = [], []
    IdXMLFile().load(str(IDXML), prot, pep)
    one = pep[:1]
    one[0].setMetaValue("spectrum_reference", "scan=does-not-exist-999999")
    bad_idxml = str(tmp_path / "bad.idXML")
    IdXMLFile().store(bad_idxml, prot, one)
    empty_mzml = str(tmp_path / "empty.mzML")
    FileHandler().storeExperiment(empty_mzml, MSExperiment())
    with pytest.raises(SpectrumRefError):
        validate_spectrum_refs(bad_idxml, empty_mzml)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py::test_validate_spectrum_refs_raises_on_mismatch -v`
Expected: FAIL with `ImportError: cannot import name 'validate_spectrum_refs'`.

- [ ] **Step 3: Implement**

```python
# append to onsite/mzid_adapter.py
from pyopenms import MSExperiment, FileHandler


class SpectrumRefError(RuntimeError):
    pass


@dataclass
class ValidationResult:
    n_total: int
    n_resolved: int
    ok: bool


def validate_spectrum_refs(idxml_path: str, mzml_path: str,
                           min_match_fraction: float = 0.5) -> ValidationResult:
    """Confirm PSM spectrum_reference values resolve to mzML spectrum native IDs.

    Raises SpectrumRefError when references exist but mostly fail to resolve
    (e.g. the mzid references spectra by a scheme that does not match the mzML),
    so the adapter never silently scores zero PSMs.
    """
    exp = MSExperiment()
    FileHandler().loadExperiment(mzml_path, exp)
    native_ids = {s.getNativeID() for s in exp.getSpectra()}

    prot, pep = [], []
    IdXMLFile().load(idxml_path, prot, pep)
    refs = [pid.getMetaValue("spectrum_reference")
            for pid in pep if pid.metaValueExists("spectrum_reference")]
    n_total = len(refs)
    n_resolved = sum(1 for r in refs if r in native_ids)

    ok = (n_total == 0) or (n_resolved / n_total >= min_match_fraction)
    if n_total > 0 and not ok:
        raise SpectrumRefError(
            f"Only {n_resolved}/{n_total} mzIdentML spectrum references resolve to "
            f"spectra in {mzml_path}. The mzIdentML likely references spectra by a "
            f"scheme that does not match the mzML native IDs. Aborting to avoid "
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

### Task 4: `run(...)` orchestration with decoy guard

**Files:**
- Modify: `onsite/mzid_adapter.py`
- Test: `tests/test_mzid_adapter.py`

**Interfaces:**
- Consumes: `mzid_to_idxml`, `idxml_to_mzid`, `validate_spectrum_refs` (Tasks 2-3); `run_all_localizers` (Task 1).
- Produces: `run(mzml_path, mzid_in, mzid_out, fragment_mass_tolerance=0.05, fragment_mass_unit="Da", threads=1, add_decoys=False, keep_intermediates=False, logger=None) -> dict` returning `{"n_psms": int, "decoy_mode": bool, "out": str}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_mzid_adapter.py
@pytest.mark.skipif(not MZML.exists(), reason="data/1.mzML not present")
def test_run_end_to_end_attaches_three_scores(tmp_path):
    from onsite.mzid_adapter import run
    from pyopenms import MzIdentMLFile
    _, mzid, _ = _make_small_mzid(tmp_path, n=5)
    out_mzid = str(tmp_path / "scored.mzid")
    result = run(str(MZML), mzid, out_mzid, threads=1)
    assert result["decoy_mode"] is False
    prot, pep = [], []
    MzIdentMLFile().load(out_mzid, prot, pep)
    assert _has_score(pep, "AScore_site_scores")
    assert _has_score(pep, "PhosphoRS_site_probs")
    assert _has_score(pep, "Luciphor_site_scores")


@pytest.mark.skipif(not MZML.exists(), reason="data/1.mzML not present")
def test_run_add_decoys_falls_back_when_no_ala(tmp_path, monkeypatch):
    from onsite import mzid_adapter
    from onsite.mzid_adapter import run, LoadInfo
    # Force has_ala False to exercise the fallback branch deterministically.
    real = mzid_adapter.mzid_to_idxml
    def fake(mzid_path, idxml_path):
        info = real(mzid_path, idxml_path)
        return LoadInfo(n_psms=info.n_psms, has_ala=False)
    monkeypatch.setattr(mzid_adapter, "mzid_to_idxml", fake)
    _, mzid, _ = _make_small_mzid(tmp_path, n=3)
    out_mzid = str(tmp_path / "scored.mzid")
    result = run(str(MZML), mzid, out_mzid, add_decoys=True, threads=1)
    assert result["decoy_mode"] is False  # fell back
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py -k "end_to_end or falls_back" -v`
Expected: FAIL with `ImportError: cannot import name 'run'` (or SKIP without mzML).

- [ ] **Step 3: Implement**

```python
# append to onsite/mzid_adapter.py
import logging
import os
import tempfile


def run(mzml_path, mzid_in, mzid_out,
        fragment_mass_tolerance=0.05, fragment_mass_unit="Da",
        threads=1, add_decoys=False, keep_intermediates=False, logger=None):
    """Score an mzIdentML+mzML pair with all three localizers; write scored mzid.

    Decoy mode runs only when add_decoys is set AND the input contains Alanine
    candidates; otherwise it falls back to the no-decoy pack with a warning.
    """
    from onsite.onsitec import run_all_localizers
    log = logger or logging.getLogger("onsite.mzid")

    tmpdir = tempfile.mkdtemp(prefix="onsite_mzid_")
    try:
        tmp_in = os.path.join(tmpdir, "input.idXML")
        merged = os.path.join(tmpdir, "merged.idXML")

        info = mzid_to_idxml(mzid_in, tmp_in)
        validate_spectrum_refs(tmp_in, mzml_path)

        decoy_mode = bool(add_decoys and info.has_ala)
        if add_decoys and not info.has_ala:
            log.warning(
                "--add-decoys was set but no Alanine candidate was found in the "
                "input; computing the no-decoy score pack instead."
            )

        run_all_localizers(
            mzml_path, tmp_in, merged,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_mass_unit=fragment_mass_unit,
            threads=threads, add_decoys=decoy_mode,
        )

        idxml_to_mzid(merged, mzid_out)
        return {"n_psms": info.n_psms, "decoy_mode": decoy_mode, "out": mzid_out}
    finally:
        if keep_intermediates:
            log.info("Intermediates kept in %s", tmpdir)
        else:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mzid_adapter.py -k "end_to_end or falls_back" -v`
Expected: PASS (or SKIP without mzML).

- [ ] **Step 5: Commit**

```bash
git add onsite/mzid_adapter.py tests/test_mzid_adapter.py
git commit -m "feat(mzid): run() orchestration with data-gated decoy fallback"
```

---

### Task 5: `onsite mzid` CLI command + console script

**Files:**
- Modify: `onsite/onsitec.py` (register command)
- Modify: `pyproject.toml` (console script)
- Test: `tests/test_mzid_adapter.py`

**Interfaces:**
- Consumes: `onsite.mzid_adapter.run` (Task 4).
- Produces: click command `mzid` registered on the `cli` group; `onsite-mzid` console entry → `onsite.onsitec:main` (the group already dispatches subcommands).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_mzid_adapter.py
def test_cli_mzid_help():
    from click.testing import CliRunner
    from onsite.onsitec import cli
    result = CliRunner().invoke(cli, ["mzid", "--help"])
    assert result.exit_code == 0
    assert "mzid" in result.output.lower()
    assert "--add-decoys" in result.output
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_mzid_adapter.py::test_cli_mzid_help -v`
Expected: FAIL — `mzid` is not a known command (exit_code != 0).

- [ ] **Step 3: Implement the command and register it**

In `onsite/onsitec.py`, add after the `all` command definition:

```python
@click.command()
@click.option("-in", "--in-file", "in_file", required=True,
              help="Input mzML file path", type=click.Path(exists=True))
@click.option("-id", "--id-file", "id_file", required=True,
              help="Input mzIdentML file path", type=click.Path(exists=True))
@click.option("-out", "--out-file", "out_file", required=True,
              help="Output mzIdentML file path", type=click.Path())
@click.option("--fragment-mass-tolerance", "fragment_mass_tolerance", type=float,
              default=0.05, help="Fragment mass tolerance (default: 0.05)")
@click.option("--fragment-mass-unit", "fragment_mass_unit",
              type=click.Choice(["Da", "ppm"]), default="Da",
              help="Tolerance unit (default: Da)")
@click.option("--threads", "threads", type=int, default=1,
              help="Number of parallel threads (default: 1)")
@click.option("--add-decoys", "add_decoys", is_flag=True, default=False,
              help="Compute the Alanine (PhosphoDecoy) score pack when Ala is present")
@click.option("--keep-intermediates", "keep_intermediates", is_flag=True, default=False,
              help="Keep temporary idXML intermediates for debugging")
def mzid(in_file, id_file, out_file, fragment_mass_tolerance, fragment_mass_unit,
         threads, add_decoys, keep_intermediates):
    """Compute AScore + PhosphoRS + LucXor on an mzIdentML+mzML pair and write a scored mzIdentML."""
    from onsite.mzid_adapter import run, SpectrumRefError
    try:
        result = run(
            in_file, id_file, out_file,
            fragment_mass_tolerance=fragment_mass_tolerance,
            fragment_mass_unit=fragment_mass_unit,
            threads=threads, add_decoys=add_decoys,
            keep_intermediates=keep_intermediates,
        )
        click.echo(f"Scored {result['n_psms']} PSMs (decoy_mode={result['decoy_mode']})")
        click.echo(f"Output: {os.path.abspath(result['out'])}")
    except SpectrumRefError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)
```

Then register it next to the other commands (near `cli.add_command(lucxor)`):

```python
cli.add_command(mzid)
```

In `pyproject.toml` under `[project.scripts]`, add:

```toml
onsite-mzid = "onsite.onsitec:main"
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_mzid_adapter.py::test_cli_mzid_help -v`
Expected: PASS.

- [ ] **Step 5: Full suite + commit**

Run: `python -m pytest tests/test_mzid_adapter.py tests/test_cli.py -v`
Expected: all PASS (mzML-dependent tests may SKIP).

```bash
git add onsite/onsitec.py pyproject.toml tests/test_mzid_adapter.py
git commit -m "feat(mzid): add 'onsite mzid' command and console script"
```

---

## Self-Review

- **Spec coverage:** Interface (Task 5), components mzid_to_idxml/idxml_to_mzid (Task 2), validate_spectrum_refs (Task 3), run() (Task 4), run_all_localizers refactor (Task 1), decoy gate + fallback (Task 4), pyOpenMS-regenerated output (Task 2/4), error handling (Tasks 3-5), testing items 1-5 (covered across tasks). Equivalence test (spec test #3) — add as an optional extra assertion in Task 4's end-to-end test comparing one PSM's `AScore_site_scores` against a direct `run_all_localizers` run if exact comparison is desired; not required for the gate.
- **Placeholders:** none — all steps carry full code/commands.
- **Type consistency:** `LoadInfo(n_psms, has_ala)`, `ValidationResult(n_total, n_resolved, ok)`, `SpectrumRefError`, `run(...) -> dict{n_psms,decoy_mode,out}`, `run_all_localizers(...)` signature used identically across Tasks 1 and 4.
