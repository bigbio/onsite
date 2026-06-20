# Design: mzIdentML adapter for the three phospho-localizers

Date: 2026-06-20
Status: Approved (design); pending implementation plan
Branch: `fix/phosphors-flr-site-prob-ranking`

## Goal

Provide a simple adapter that, given an **mzIdentML** identification file and its
**mzML** spectra, computes all three phosphosite-localization scores (AScore,
PhosphoRS, LucXor) and writes a new mzIdentML carrying those scores. Support a
no-decoy mode (default) and an Alanine-decoy mode (`PhosphoDecoy`) gated by a
flag **and** by Alanine being present in the data.

Today the entire pipeline (`onsite all` in `onsite/onsitec.py`) operates on
**mzML + idXML** and attaches scores as `PeptideHit` MetaValues
(`AScore_site_scores`, `PhosphoRS_site_probs`, `PhosphoRS_site_delta`,
`Luciphor_site_scores`). There is no mzIdentML support anywhere yet.

## Approach (chosen: convert-and-wrap)

A thin adapter bridges mzIdentML to the existing, tested idXML pipeline. No new
scoring logic; `pyOpenMS` `MzIdentMLFile` ↔ `IdXMLFile` produce the same OpenMS
objects, so scores attached as MetaValues serialize to mzIdentML `userParam`s
automatically.

Rejected alternative — "native in-process" (load mzid, call the three scoring
cores directly): cleaner pipeline but requires refactoring LucXor's dataset-wide
model-training orchestration to run on a `pep_ids` list. More code/risk than a
"simple adapter" warrants.

The output mzIdentML is **pyOpenMS-regenerated** (via `MzIdentMLFile.store`): it
carries the identifications + the three score `userParam`s, but does not
byte-preserve unrelated sections of a third-party source mzid. (Accepted.)

## Interface

New subcommand `onsite mzid` (registered in `onsitec.py`; optional `onsite-mzid`
console script in `pyproject.toml`):

```
onsite mzid -in spectra.mzML -id input.mzid -out scored.mzid \
            [--fragment-mass-tolerance 0.05] [--fragment-mass-unit Da|ppm] \
            [--threads 1] [--add-decoys] [--keep-intermediates]
```

Option names/defaults mirror the existing `ascore`/`all` CLIs for consistency.

## Components

New module `onsite/mzid_adapter.py` — single purpose: format bridge + orchestration.

- `mzid_to_idxml(mzid_path, idxml_path) -> LoadInfo`
  `MzIdentMLFile().load` → `IdXMLFile().store`. `LoadInfo` reports `n_psms` and
  `has_ala` (≥1 analyzed peptide contains an `A` residue — the practical
  condition for decoy-mode scoring to have any effect).
- `idxml_to_mzid(idxml_path, mzid_path)`
  `IdXMLFile().load` → `MzIdentMLFile().store`.
- `validate_spectrum_refs(pep_ids, mzml_path) -> ValidationResult`
  Build the set of mzML spectrum native IDs; confirm the PSM `spectrum_reference`
  values resolve. If a large fraction do not resolve, fail loudly (the mzid likely
  references spectra by a scheme that does not match the mzML native IDs).
- `run(mzml, mzid_in, mzid_out, tolerance, unit, threads, add_decoys, keep_intermediates)`
  Orchestrates the three steps in a `TemporaryDirectory`.

Minor refactor: extract the body of the existing `all` command into
`run_all_localizers(mzml, idxml, out, tolerance, unit, threads, add_decoys)` in
`onsitec.py`, called by both `all` (unchanged behavior) and the new `mzid`
command.

## Data flow

```
input.mzid + spectra.mzML
  └─ mzid_to_idxml ─▶ tmp_in.idXML
       └─ validate_spectrum_refs(tmp_in pep_ids, spectra.mzML)
            └─ run_all_localizers(spectra.mzML, tmp_in.idXML, add_decoys) ─▶ merged.idXML
                 │   (AScore + PhosphoRS + LucXor; scores as PeptideHit MetaValues)
                 └─ idxml_to_mzid ─▶ scored.mzid   (scores as userParams)
```

## Decoy behavior

- Default (no `--add-decoys`): no-decoy score pack (S/T/Y candidates only).
- `--add-decoys` AND `has_ala` true: decoy-mode pack (A scored as `PhosphoDecoy`
  acceptor). One mode per output file.
- `--add-decoys` set but `has_ala` false: emit a clear warning and fall back to
  the no-decoy pack (do **not** fail the run).

## Error handling

- Missing/unreadable mzid or mzML → clear, actionable error.
- mzid parse failure → surface the pyOpenMS error with context.
- Spectrum-reference mismatch → fail loudly with guidance (never silently score
  zero PSMs).
- Intermediates live in a `TemporaryDirectory` (auto-cleaned); `--keep-intermediates`
  retains them (path logged) for debugging.

## Testing (`tests/test_mzid_adapter.py`)

Fixture: generate a small mzid by round-tripping `data/1_consensus_fdr_filter_pep.idXML`
→ mzid (`IdXMLFile.load` → `MzIdentMLFile.store`).

1. Output mzid contains all three score `userParam`s on scored PSMs.
2. `--add-decoys` produces decoy-mode scores when Ala present; warns + falls back
   to no-decoy pack when no Ala present.
3. Equivalence: scores in the output mzid match running `onsite all` directly on
   the idXML (same PSMs, same score values).
4. `validate_spectrum_refs` fails loudly on a deliberately mismatched mzid.
5. `mzid_to_idxml` / `idxml_to_mzid` round-trip preserves PSM count and sequences.

## Out of scope (YAGNI)

- Making every existing CLI format-agnostic (idXML *or* mzid by extension).
- Byte-preserving in-place XML editing of a source mzid.
- Emitting both decoy and no-decoy score sets in a single file (one mode per run).
