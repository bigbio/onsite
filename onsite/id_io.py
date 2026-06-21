#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified identification I/O — Phase 1 + Phase 2: load and save side.

Provides:
    detect_format(path)          -> 'idparquet' | 'idxml' | 'mzid'
    load_identifications(path)   -> (psms_df, proteins_df, search_params_df, protein_groups_df)
    save_identifications(path, psms_df, proteins_df, template_df, source_idparquet)

Internal helpers:
    _peptide_ids_to_psms_df(prot, pep, source_path) -> pd.DataFrame
    _psms_df_to_peptide_ids(psms_df, proteins_df)   -> (prot_list, PeptideIdentificationList)
"""

import os
import re
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path) -> str:
    """Return the file format tag for *path*.

    Parameters
    ----------
    path : str or pathlib.Path

    Returns
    -------
    'idparquet'  – path is a directory, or ends with ``.idparquet``
    'mzid'       – path (lowercased) ends with ``.mzid`` or ``.mzidentml``
    'idxml'      – path ends with ``.idxml``

    Raises
    ------
    ValueError   – when no pattern matches.
    """
    path = str(path)  # coerce pathlib.Path → str
    lpath = path.lower()
    if os.path.isdir(path) or lpath.endswith(".idparquet"):
        return "idparquet"
    if lpath.endswith(".mzid") or lpath.endswith(".mzidentml"):
        return "mzid"
    if lpath.endswith(".idxml"):
        return "idxml"
    raise ValueError(
        f"Cannot detect identification format for path: {path!r}. "
        "Expected a directory/.idparquet, a .mzid/.mzidentml, or an .idxml file."
    )


# ---------------------------------------------------------------------------
# Public load entry-point
# ---------------------------------------------------------------------------

def load_identifications(
    path: str,
) -> tuple:
    """Load identifications from *path*, returning a 4-tuple of DataFrames.

    Returns
    -------
    (psms_df, proteins_df, search_params_df, protein_groups_df)

    The tuple shape mirrors ``onsite.idparquet.load_dataframes``.
    """
    fmt = detect_format(path)

    if fmt == "idparquet":
        from onsite.idparquet import load_dataframes
        return load_dataframes(path)

    if fmt == "idxml":
        return _load_idxml(path)

    if fmt == "mzid":
        return _load_mzid(path)

    raise ValueError(f"Unsupported format: {fmt!r}")  # unreachable


# ---------------------------------------------------------------------------
# Public save entry-point
# ---------------------------------------------------------------------------

def save_identifications(
    path: str,
    psms_df: pd.DataFrame,
    proteins_df=None,
    template_df=None,
    source_idparquet=None,
) -> str:
    """Save identifications to *path*, dispatching on file extension.

    Parameters
    ----------
    path : str
        Output path. Extension determines format:
        ``.idparquet`` or directory → idParquet
        ``.idxml`` → idXML
        ``.mzid`` / ``.mzidentml`` → mzIdentML
    psms_df : pd.DataFrame
        PSMs DataFrame with the 29-column schema.
    proteins_df : pd.DataFrame, optional
        Proteins DataFrame.
    template_df : pd.DataFrame, optional
        Template DataFrame (idParquet only, forwarded to save_dataframes).
    source_idparquet : str, optional
        Path to source idParquet directory. When provided together with
        idParquet output, ``save_dataframes`` is used (schema-copy mode).
        Otherwise, ``save_psms_from_scratch`` is used.

    Returns
    -------
    str
        The output path (idParquet returns the directory path).
    """
    path = str(path)
    try:
        fmt = detect_format(path)
    except ValueError:
        # For extensionless paths (e.g. "-out results"), treat as idparquet
        if os.path.splitext(path)[1] == "":
            fmt = "idparquet"
        else:
            raise

    if fmt == "idparquet":
        if source_idparquet is not None and (
            os.path.isdir(source_idparquet)
            or str(source_idparquet).endswith(".idparquet")
        ):
            from onsite.idparquet import save_dataframes
            return save_dataframes(
                path, psms_df, proteins_df,
                template_df=template_df,
                source_idparquet=source_idparquet,
            )
        else:
            # TODO(phase3): forward template_df to save_psms_from_scratch once the
            # parameter wiring (search_params.parquet population) is implemented.
            from onsite.idparquet import save_psms_from_scratch
            return save_psms_from_scratch(path, psms_df, proteins_df)

    if fmt == "idxml":
        prot, pep = _psms_df_to_peptide_ids(psms_df, proteins_df)
        import pyopenms as oms
        oms.IdXMLFile().store(str(path), prot, pep)
        return path

    if fmt == "mzid":
        prot, pep = _psms_df_to_peptide_ids(psms_df, proteins_df)
        # Strip non-UNIMOD / custom mods from SearchParameters to avoid mzid errors
        import pyopenms as oms
        for pi in prot:
            sp = pi.getSearchParameters()
            vmods = sp.variable_modifications
            filtered = [m for m in vmods if b"PhosphoDecoy" not in m]
            sp.variable_modifications = filtered
            pi.setSearchParameters(sp)
        oms.MzIdentMLFile().store(str(path), prot, pep)
        return path

    raise ValueError(f"Unsupported format: {fmt!r}")  # unreachable


# ---------------------------------------------------------------------------
# idXML / mzid loaders
# ---------------------------------------------------------------------------

def _load_idxml(path: str) -> tuple:
    """Load an idXML file via pyOpenMS and convert to DataFrames."""
    import pyopenms as oms

    prot: List = []
    pep = oms.PeptideIdentificationList()
    oms.IdXMLFile().load(path, prot, pep)
    logger.info("Loaded %d PeptideIdentifications from %s", pep.size(), path)

    psms_df = _peptide_ids_to_psms_df(prot, pep, path)
    proteins_df = _prot_ids_to_df(prot)
    return psms_df, proteins_df, pd.DataFrame(), pd.DataFrame()


def _load_mzid(path: str) -> tuple:
    """Load an mzIdentML file via pyOpenMS and convert to DataFrames."""
    import pyopenms as oms

    prot: List = []
    pep = oms.PeptideIdentificationList()
    oms.MzIdentMLFile().load(path, prot, pep)
    logger.info("Loaded %d PeptideIdentifications from %s", pep.size(), path)

    psms_df = _peptide_ids_to_psms_df(prot, pep, path)
    proteins_df = _prot_ids_to_df(prot)
    return psms_df, proteins_df, pd.DataFrame(), pd.DataFrame()


# ---------------------------------------------------------------------------
# _peptide_ids_to_psms_df
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Keys that are never used as a fallback score candidate.
_NON_SCORE_KEYS = {
    "calcMZ",
    "pass_threshold",
    "target_decoy",
    "AScore_site_scores",
    "num_matched_peptides",
    "protein_references",
    "isotope_error",
}

# Scan regex — require a word boundary so "basescan=123" does not match
_SCAN_RE = re.compile(r"\bscan=(\d+)")


def _peptide_ids_to_psms_df(prot, pep, source_path: str) -> pd.DataFrame:
    """Convert pyOpenMS PeptideIdentification objects to a psms DataFrame.

    Parameters
    ----------
    prot : list[ProteinIdentification]
        Protein identification list (may be empty).
    pep : PeptideIdentificationList or iterable[PeptideIdentification]
        Peptide identification list.
    source_path : str
        Original file path — used to populate ``reference_file_name``.

    Returns
    -------
    pd.DataFrame
        Schema-compatible with the psms.parquet fixture (29 columns).
    """
    from onsite.idparquet import pyopenms_to_unimod_notation, peptidoform_to_modifications

    reference_file_name = os.path.basename(source_path)
    rows: List[dict] = []

    for pid_idx, pid in enumerate(pep):
        score_type = pid.getScoreType()
        higher_score_better = pid.isHigherScoreBetter()

        rt_val = pid.getRT()
        rt = rt_val if not _is_sentinel(rt_val) else float("nan")

        mz_val = pid.getMZ()
        observed_mz = mz_val if not _is_sentinel(mz_val) else float("nan")

        # spectrum_reference
        spectrum_reference = ""
        if pid.metaValueExists("spectrum_reference"):
            spectrum_reference = str(pid.getMetaValue("spectrum_reference"))

        # scan
        scan_match = _SCAN_RE.search(spectrum_reference)
        scan = np.int32(int(scan_match.group(1))) if scan_match else np.int32(-1)

        hits = pid.getHits()
        for hit_idx, hit in enumerate(hits):
            seq_obj = hit.getSequence()
            sequence = seq_obj.toUnmodifiedString()
            peptidoform = pyopenms_to_unimod_notation(seq_obj.toString())
            precursor_charge = np.int32(hit.getCharge())
            score = float(hit.getScore())

            # is_decoy
            is_decoy = False
            if hit.metaValueExists("target_decoy"):
                td = str(hit.getMetaValue("target_decoy"))
                is_decoy = td.strip().lower() == "decoy"

            # protein_accessions: numpy object array of dicts matching fixture schema
            # fixture element: {'accession': ..., 'aa_before': ..., 'aa_after': ...,
            #                   'start': ..., 'end': ...}
            pa_list = []
            for ev in hit.getPeptideEvidences():
                pa_list.append({
                    "accession": ev.getProteinAccession(),
                    "aa_before": str(ev.getAABefore()),
                    "aa_after": str(ev.getAAAfter()),
                    "start": ev.getStart(),
                    "end": ev.getEnd(),
                })
            protein_accessions = np.array(pa_list, dtype=object)

            # psm_metavalues: numpy object array of dicts with correct value_type
            keys: List = []
            hit.getKeys(keys)
            meta_list = []
            meta_by_name: dict = {}
            for k in keys:
                k_str = k.decode() if isinstance(k, bytes) else str(k)
                try:
                    raw_val = hit.getMetaValue(k)
                    v_type = _metavalue_type_str(raw_val)
                    v_str = str(raw_val)
                except Exception:
                    raw_val = None
                    v_type = "string"
                    v_str = ""
                meta_list.append({"name": k_str, "value": v_str, "value_type": v_type})
                meta_by_name[k_str] = v_str
            psm_metavalues = np.array(meta_list, dtype=object)

            # mzIdentML round-trip can rename the score_type (e.g. to
            # "PSM-level search engine specific statistic") and move the
            # original score value into a named meta value. When hit.getScore()
            # is 0.0, try to recover the score ONLY from a metavalue whose
            # name exactly matches score_type. A legitimate 0.0 score must
            # not be overwritten by unrelated numeric metavalues (e.g. counts).
            if score == 0.0:
                if score_type in meta_by_name:
                    try:
                        score = float(meta_by_name[score_type])
                    except (ValueError, TypeError):
                        pass

            # modifications: populated from the peptidoform
            modifications = peptidoform_to_modifications(peptidoform)

            rows.append({
                # --- core columns ---
                "sequence": sequence,
                "peptidoform": peptidoform,
                "modifications": modifications,
                "precursor_charge": precursor_charge,
                "posterior_error_probability": float("nan"),
                "is_decoy": is_decoy,
                "calculated_mz": float("nan"),
                "observed_mz": observed_mz,
                "additional_scores": np.array([], dtype=object),
                "protein_accessions": protein_accessions,
                "predicted_rt": float("nan"),
                "reference_file_name": reference_file_name,
                "cv_params": None,
                "scan": scan,
                "rt": rt,
                "ion_mobility": float("nan"),
                "spectrum_reference": spectrum_reference,
                "score": score,
                "score_type": score_type,
                "higher_score_better": higher_score_better,
                "hit_index": np.int32(hit_idx),
                "peptide_identification_index": np.int32(pid_idx),
                "psm_metavalues": psm_metavalues,
                "spectrum_metavalues": np.array([], dtype=object),
                "run_identifier": None,
                "mz_array": None,
                "intensity_array": None,
                "charge_array": None,
                "ion_type_array": None,
            })

    if not rows:
        return _empty_psms_df()

    df = pd.DataFrame(rows)

    # Enforce dtypes that must match the fixture schema
    df["precursor_charge"] = df["precursor_charge"].astype("int32")
    df["scan"] = df["scan"].astype("int32")
    df["hit_index"] = df["hit_index"].astype("int32")
    df["peptide_identification_index"] = df["peptide_identification_index"].astype("int32")
    df["score"] = df["score"].astype("float64")
    df["observed_mz"] = df["observed_mz"].astype("float64")
    df["rt"] = df["rt"].astype("float64")
    df["is_decoy"] = df["is_decoy"].astype("bool")
    df["posterior_error_probability"] = df["posterior_error_probability"].astype("float64")
    df["calculated_mz"] = df["calculated_mz"].astype("float64")
    df["predicted_rt"] = df["predicted_rt"].astype("float64")
    df["ion_mobility"] = df["ion_mobility"].astype("float64")

    return df


# ---------------------------------------------------------------------------
# _psms_df_to_peptide_ids  (Phase 2 — save side)
# ---------------------------------------------------------------------------

def _psms_df_to_peptide_ids(psms_df: pd.DataFrame, proteins_df=None):
    """Convert a psms DataFrame to pyOpenMS PeptideIdentification objects.

    Parameters
    ----------
    psms_df : pd.DataFrame
        PSMs DataFrame with the 29-column schema.
    proteins_df : pd.DataFrame, optional
        Proteins DataFrame (accession column used if present).

    Returns
    -------
    (prot_list, PeptideIdentificationList)
    """
    import pyopenms as oms
    from onsite.idparquet import unimod_to_pyopenms_notation

    pep_list = oms.PeptideIdentificationList()

    # Group by peptide_identification_index
    group_col = "peptide_identification_index"
    if group_col not in psms_df.columns:
        psms_df = psms_df.copy()
        psms_df[group_col] = 0

    for _, group in psms_df.groupby(group_col, sort=True):
        first = group.iloc[0]

        # score_type — use pd.isna() to catch None, NaN, and pd.NA
        score_type = first.get("score_type", None)
        if pd.isna(score_type):
            score_type = "score"

        # higher_score_better — use pd.isna() to catch None, NaN, and pd.NA
        hsb_val = first.get("higher_score_better", False)
        if pd.isna(hsb_val):
            hsb_val = False
        higher_score_better = bool(hsb_val)

        pid = oms.PeptideIdentification()
        pid.setScoreType(score_type)
        pid.setHigherScoreBetter(higher_score_better)
        pid.setIdentifier("run1")

        rt_val = first.get("rt", float("nan"))
        # Guard against None, NaN, pd.NA, and non-finite values
        if not pd.isna(rt_val) and np.isfinite(float(rt_val)):
            pid.setRT(float(rt_val))

        mz_val = first.get("observed_mz", float("nan"))
        if not pd.isna(mz_val) and np.isfinite(float(mz_val)):
            pid.setMZ(float(mz_val))

        spec_ref = first.get("spectrum_reference", None)
        if spec_ref and str(spec_ref).strip():
            pid.setMetaValue("spectrum_reference", str(spec_ref))

        # Build hits sorted by hit_index
        hits = []
        sort_col = "hit_index" if "hit_index" in group.columns else None
        sorted_group = group.sort_values("hit_index") if sort_col else group

        for _, row in sorted_group.iterrows():
            pf = row.get("peptidoform", None)
            if pf is None or (isinstance(pf, float) and np.isnan(pf)):
                logger.warning("Skipping hit with None/NaN peptidoform at index %s", row.name)
                continue
            pf_str = str(pf)

            try:
                pyo_seq = unimod_to_pyopenms_notation(pf_str)
                seq = oms.AASequence.fromString(pyo_seq)
            except Exception as exc:
                logger.warning("Could not parse peptidoform %r: %s", pf_str, exc)
                continue

            hit = oms.PeptideHit()
            hit.setSequence(seq)

            charge = row.get("precursor_charge", 0)
            # Guard against None and pd.NA (pd.isna covers all NA sentinels)
            hit.setCharge(int(charge) if not pd.isna(charge) else 0)

            score_val = row.get("score", 0.0)
            final_score = float(score_val) if not pd.isna(score_val) else 0.0
            hit.setScore(final_score)

            # Persist the score under the score_type name as a metavalue so
            # that the score-recovery path (score_type in meta_by_name) can
            # find it after an mzIdentML round-trip, where the score_type is
            # renamed by pyOpenMS to "PSM-level search engine specific statistic"
            # and the original score is stored under the original score_type name.
            if score_type and score_type not in ("", "score"):
                hit.setMetaValue(score_type, final_score)

            # Restore is_decoy as target_decoy metavalue (only if not already
            # present in psm_metavalues, to avoid duplicate entries)
            is_decoy_val = row.get("is_decoy", None)
            if is_decoy_val is not None and not pd.isna(is_decoy_val):
                td_str = "decoy" if bool(is_decoy_val) else "target"
                hit.setMetaValue("target_decoy", td_str)

            # Restore protein_accessions as PeptideEvidence objects.
            # protein_accessions is stored as a numpy object array of dicts:
            #   {'accession': str, 'aa_before': str, 'aa_after': str,
            #    'start': int, 'end': int}
            pa_val = row.get("protein_accessions", None)
            if pa_val is not None:
                if isinstance(pa_val, np.ndarray):
                    pa_iter = pa_val.tolist()
                elif isinstance(pa_val, list):
                    pa_iter = pa_val
                else:
                    pa_iter = []
                evidences = []
                for entry in pa_iter:
                    if isinstance(entry, dict):
                        ev = oms.PeptideEvidence()
                        acc = entry.get("accession", "")
                        if acc:
                            ev.setProteinAccession(str(acc))
                        aa_before = entry.get("aa_before", "")
                        aa_after = entry.get("aa_after", "")
                        if aa_before:
                            ev.setAABefore(aa_before.encode() if isinstance(aa_before, str) else aa_before)
                        if aa_after:
                            ev.setAAAfter(aa_after.encode() if isinstance(aa_after, str) else aa_after)
                        start = entry.get("start", -1)
                        end = entry.get("end", -1)
                        try:
                            ev.setStart(int(start))
                            ev.setEnd(int(end))
                        except (ValueError, TypeError):
                            pass
                        evidences.append(ev)
                    elif isinstance(entry, str):
                        # Fallback: plain accession string
                        ev = oms.PeptideEvidence()
                        ev.setProteinAccession(entry)
                        evidences.append(ev)
                if evidences:
                    hit.setPeptideEvidences(evidences)

            # Set metavalues
            mv = row.get("psm_metavalues", None)
            if mv is not None:
                # handle numpy array, list, or None
                if isinstance(mv, np.ndarray):
                    mv_iter = mv.tolist() if mv.ndim > 0 else []
                elif isinstance(mv, list):
                    mv_iter = mv
                else:
                    mv_iter = []
                for entry in mv_iter:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name", "")
                    value = entry.get("value", "")
                    vtype = entry.get("value_type", "string")
                    if not name:
                        continue
                    try:
                        if vtype == "double":
                            hit.setMetaValue(name, float(value))
                        elif vtype == "int":
                            hit.setMetaValue(name, int(float(value)))
                        else:
                            hit.setMetaValue(name, str(value))
                    except (ValueError, TypeError):
                        hit.setMetaValue(name, str(value))

            hits.append(hit)

        pid.setHits(hits)
        pep_list.push_back(pid)

    # Build ProteinIdentification
    pi = oms.ProteinIdentification()
    pi.setIdentifier("run1")

    # Collect unique score_type for SearchParameters
    score_types = psms_df["score_type"].dropna().unique().tolist() if "score_type" in psms_df.columns else []
    score_type_main = score_types[0] if score_types else "score"
    pi.setScoreType(score_type_main)

    higher_score_better_main = False
    if "higher_score_better" in psms_df.columns:
        vals = psms_df["higher_score_better"].dropna().unique().tolist()
        if vals:
            higher_score_better_main = bool(vals[0])
    pi.setHigherScoreBetter(higher_score_better_main)

    sp = pi.getSearchParameters()

    # Collect UNIMOD mods seen in peptidoforms and add as variable modifications
    seen_mods = set()
    if "peptidoform" in psms_df.columns:
        for pf in psms_df["peptidoform"].dropna():
            for m in re.finditer(r"\[UNIMOD:(\d+)\]", str(pf)):
                seen_mods.add(f"UNIMOD:{m.group(1)}")

    # Only add standard UNIMOD mods (skip PhosphoDecoy or custom names)
    vmods = []
    for unimod_acc in sorted(seen_mods):
        # Convert UNIMOD accession to pyOpenMS name for use in SearchParameters
        # Use a dummy single-AA peptidoform to get the mod name
        test_pf = f"S[{unimod_acc}]"
        pyo_name = unimod_to_pyopenms_notation(test_pf)
        # Extract just the modification name from "S(Name)" format
        m = re.search(r"\(([^)]+)\)", pyo_name)
        if m:
            mod_name = m.group(1)
            # Skip custom / non-standard mods
            if b"PhosphoDecoy" not in mod_name.encode():
                vmods.append(mod_name.encode())

    sp.variable_modifications = vmods
    pi.setSearchParameters(sp)

    # Collect protein accessions from proteins_df AND from protein_accessions
    # column in psms_df (populated on load from PeptideEvidence objects).
    # idXML requires every accession referenced in PeptideEvidence to be
    # registered as a ProteinHit in the ProteinIdentification.
    seen_accs: set = set()
    prot_hits = []
    if proteins_df is not None and "accession" in proteins_df.columns:
        for acc in proteins_df["accession"].dropna():
            acc_str = str(acc)
            if acc_str not in seen_accs:
                ph = oms.ProteinHit()
                ph.setAccession(acc_str)
                prot_hits.append(ph)
                seen_accs.add(acc_str)
    # Also register any accession found in psm_metavalues / protein_accessions
    if "protein_accessions" in psms_df.columns:
        for pa_val in psms_df["protein_accessions"].dropna():
            if isinstance(pa_val, np.ndarray):
                pa_iter = pa_val.tolist()
            elif isinstance(pa_val, list):
                pa_iter = pa_val
            else:
                continue
            for entry in pa_iter:
                if isinstance(entry, dict):
                    acc_str = str(entry.get("accession", ""))
                elif isinstance(entry, str):
                    acc_str = entry
                else:
                    continue
                if acc_str and acc_str not in seen_accs:
                    ph = oms.ProteinHit()
                    ph.setAccession(acc_str)
                    prot_hits.append(ph)
                    seen_accs.add(acc_str)
    pi.setHits(prot_hits)

    prot = [pi]
    return prot, pep_list


def _metavalue_type_str(value) -> str:
    """Map a Python value returned by getMetaValue() to the fixture value_type string.

    Mapping:
        float  -> "double"   (matches fixture convention)
        int    -> "int"
        other  -> "string"
    """
    if isinstance(value, float):
        return "double"
    if isinstance(value, int):
        return "int"
    return "string"


def _is_sentinel(value: float) -> bool:
    """Return True if *value* is the pyOpenMS unset-RT/MZ sentinel (-1e38 range)."""
    return value < -1e30


# ---------------------------------------------------------------------------
# proteins helper
# ---------------------------------------------------------------------------

def _prot_ids_to_df(prot) -> pd.DataFrame:
    """Build a simple proteins DataFrame from ProteinIdentification objects."""
    if not prot:
        return pd.DataFrame()

    rows = []
    for pi in prot:
        for ph in pi.getHits():
            rows.append({"accession": ph.getAccession()})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Empty DataFrame helper
# ---------------------------------------------------------------------------

_PSM_COLUMNS = [
    "sequence", "peptidoform", "modifications", "precursor_charge",
    "posterior_error_probability", "is_decoy", "calculated_mz", "observed_mz",
    "additional_scores", "protein_accessions", "predicted_rt", "reference_file_name",
    "cv_params", "scan", "rt", "ion_mobility", "spectrum_reference", "score",
    "score_type", "higher_score_better", "hit_index", "peptide_identification_index",
    "psm_metavalues", "spectrum_metavalues", "run_identifier",
    "mz_array", "intensity_array", "charge_array", "ion_type_array",
]


# dtype map for the empty DataFrame — matches the populated path and fixture schema
_PSM_DTYPE_MAP = {
    "precursor_charge": "int32",
    "scan": "int32",
    "hit_index": "int32",
    "peptide_identification_index": "int32",
    "score": "float64",
    "rt": "float64",
    "observed_mz": "float64",
    "posterior_error_probability": "float64",
    "calculated_mz": "float64",
    "predicted_rt": "float64",
    "ion_mobility": "float64",
    "is_decoy": "bool",
    "higher_score_better": "bool",
}


def _empty_psms_df() -> pd.DataFrame:
    """Return an empty DataFrame with the full psms schema and correct dtypes."""
    df = pd.DataFrame(columns=_PSM_COLUMNS)
    for col, dtype in _PSM_DTYPE_MAP.items():
        df[col] = df[col].astype(dtype)
    return df
