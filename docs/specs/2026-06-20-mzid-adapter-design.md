# Design: format-agnostic idXML / mzIdentML I/O for the three phospho-localizers

Date: 2026-06-20 (revised same day: format-agnostic instead of convert-and-wrap)
Status: Approved (design); implementation in progress
Branch: `fix/phosphors-flr-site-prob-ranking`

## Goal

Let the localization pipeline read **and** write either idXML or mzIdentML by
file extension, so `onsite all -in spectra.mzML -id ids.mzid -out scored.mzid`
computes all three scores (AScore, PhosphoRS, LucXor) and writes a scored
mzIdentML. Support a no-decoy mode (default) and an Alanine-decoy
(`PhosphoDecoy`) mode gated by a flag **and** by Alanine being present in the
data.

Scores attach as `PeptideHit` MetaValues (`AScore_site_scores`,
`PhosphoRS_site_probs`, `PhosphoRS_site_delta`, `Luciphor_site_scores`) and
serialize to mzIdentML `userParam`s automatically.

## Approach (chosen: format-agnostic I/O helpers)

`pyOpenMS` `MzIdentMLFile` and `IdXMLFile` `load`/`store` operate on the same
objects (a plain list of `ProteinIdentification` + a `PeptideIdentificationList`).
So a single pair of helpers that dispatch on file extension lets every command
accept either format with no temp-file round-trip:

- `is_mzid(path) -> bool` — true for `.mzid` / `.mzId` / `.mzIdentML` (case-insensitive).
- `load_identifications(path) -> (prot: list, pep: PeptideIdentificationList)`
  — `MzIdentMLFile().load` for mzid, else `IdXMLFile().load`. Both populate a
  `PeptideIdentificationList` (built via `push_back`); prot ids are a plain list.
- `store_identifications(path, prot, pep) -> None` — `MzIdentMLFile().store` for
  mzid, else `IdXMLFile().store`.

Rejected alternative — "convert-and-wrap" (mzid→idXML temp → pipeline →
idXML→mzid): an avoidable round-trip; the format-agnostic helpers are cleaner and
make mzid support fall out across all commands.

### Critical mzIdentML-store detail (verified)

`MzIdentMLFile().store()` raises `Invalid CV identifier!` when a search
parameter's `variable_modifications` lists a non-UNIMOD custom modification —
specifically `PhosphoDecoy (A)` → `UNIMOD:99913`. The fix, verified by probe:
**before an mzid store, strip from each `ProteinIdentification`'s
`SearchParameters.variable_modifications` any entry that is not a real UNIMOD
modification** (i.e. the `PhosphoDecoy` entries). The `PhosphoDecoy` modification
*on the peptide hits* serializes correctly and round-trips back intact
(`AS(Phospho)A(PhosphoDecoy)K` with score userParams preserved). `store_identifications`
performs this stripping for the mzid branch only; idXML is unaffected.

Output mzIdentML is pyOpenMS-regenerated (carries identifications + score
userParams; does not byte-preserve unrelated sections of a third-party source). (Accepted.)

## Components

`onsite/mzid_adapter.py` (replaces the Task 2 conversion helpers):
- `is_mzid(path)`, `load_identifications(path)`, `store_identifications(path, prot, pep)` (above).
- `has_alanine(pep) -> bool` — ≥1 peptide hit's unmodified sequence contains `A`
  (the practical precondition for decoy-mode scoring to do anything).
- `validate_spectrum_refs(pep, mzml_path, min_match_fraction=0.5) -> ValidationResult`
  — confirm PSM `spectrum_reference`s resolve to mzML spectrum native IDs; raise
  `SpectrumRefError` when references exist but mostly fail to resolve, so the
  pipeline never silently scores zero PSMs.

CLI wiring (`onsite/ascore/cli.py`, `onsite/phosphors/cli.py`,
`onsite/lucxor/cli.py`): replace hardcoded `IdXMLFile().load(id_file, …)` with
`load_identifications(id_file)` and the final `IdXMLFile().store(out, …)` with
`store_identifications(out, …)`. Each command then accepts idXML or mzid by
extension.

`onsite/onsitec.py` — `run_all_localizers` (from Task 1): before invoking the
tools, load the id file once via `load_identifications` to (a) run
`validate_spectrum_refs` against the mzML and (b) compute `has_alanine` for the
decoy guard. The intermediate per-tool files stay idXML (controlled internally);
the final merged output is written via `store_identifications`, so the output
format follows the `-out` extension. `merge_algorithm_results` writes its output
through `store_identifications`.

No separate `onsite mzid` command — `onsite all` (and each individual command)
handles both formats by extension. (Supersedes the earlier `onsite mzid` plan.)

## Decoy behavior

- Default (no `--add-decoys`): no-decoy score pack (S/T/Y candidates only).
- `--add-decoys` AND `has_alanine(pep)` true: decoy-mode pack (A as `PhosphoDecoy`).
- `--add-decoys` set but no Ala present: clear warning, fall back to no-decoy pack
  (never fail for this reason).

## Error handling

- Missing/unreadable mzid or mzML → clear error.
- Spectrum-reference mismatch → `SpectrumRefError` with guidance (never silently
  score zero PSMs).
- mzid store custom-modification quirk → handled in `store_identifications`.

## Testing (`tests/test_mzid_adapter.py`)

1. `load_identifications` / `store_identifications` round-trip preserves PSM count
   and sequences for BOTH idXML and mzid.
2. mzid store of a hit carrying `PhosphoDecoy (A)` succeeds and round-trips
   (sequence + score userParam intact), even though the modification is custom.
3. `has_alanine` true/false on crafted inputs.
4. `validate_spectrum_refs` raises `SpectrumRefError` on a deliberately mismatched
   mzid; passes on matching native IDs.
5. End-to-end (skip if `data/1.mzML` absent): `onsite all -id <mzid> -out <mzid>`
   produces an mzid carrying all three score userParams; `--add-decoys` with no Ala
   warns and falls back.

## Out of scope (YAGNI)

- Byte-preserving in-place XML editing of a source mzid.
- Emitting both decoy and no-decoy score sets in one file (one mode per run).
