#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified identification I/O — Phase 1: load side.

Provides:
    detect_format(path)          -> 'idparquet' | 'idxml' | 'mzid'
    load_identifications(path)   -> (psms_df, proteins_df, search_params_df, protein_groups_df)

Internal helper:
    _peptide_ids_to_psms_df(prot, pep, source_path) -> pd.DataFrame
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

def detect_format(path: str) -> str:
    """Return the file format tag for *path*.

    Returns
    -------
    'idparquet'  – path is a directory, or ends with ``.idparquet``
    'mzid'       – path (lowercased) ends with ``.mzid`` or ``.mzidentml``
    'idxml'      – path ends with ``.idxml``

    Raises
    ------
    ValueError   – when no pattern matches.
    """
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

_SCAN_RE = re.compile(r"scan=(\d+)")


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
    from onsite.idparquet import pyopenms_to_unimod_notation

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

            # psm_metavalues: numpy object array of dicts
            keys: List = []
            hit.getKeys(keys)
            meta_list = []
            meta_by_name: dict = {}
            for k in keys:
                k_str = k.decode() if isinstance(k, bytes) else str(k)
                try:
                    v_str = str(hit.getMetaValue(k))
                except Exception:
                    v_str = ""
                meta_list.append({"name": k_str, "value": v_str, "value_type": "string"})
                meta_by_name[k_str] = v_str
            psm_metavalues = np.array(meta_list, dtype=object)

            # mzIdentML round-trip can rename the score_type (e.g. to
            # "PSM-level search engine specific statistic") and move the
            # original score value into a named meta value. When hit.getScore()
            # is 0.0, try to recover the score from:
            #  1. A metavalue whose name matches score_type (idXML round-trip)
            #  2. Any numeric metavalue that is not a known non-score field
            _NON_SCORE_KEYS = {"calcMZ", "pass_threshold", "target_decoy"}
            if score == 0.0:
                if score_type in meta_by_name:
                    try:
                        score = float(meta_by_name[score_type])
                    except (ValueError, TypeError):
                        pass
                if score == 0.0:
                    for _k, _v in meta_by_name.items():
                        if _k in _NON_SCORE_KEYS:
                            continue
                        try:
                            candidate = float(_v)
                        except (ValueError, TypeError):
                            continue
                        if candidate != 0.0:
                            score = candidate
                            break

            rows.append({
                # --- core columns ---
                "sequence": sequence,
                "peptidoform": peptidoform,
                "modifications": np.array([], dtype=object),
                "precursor_charge": precursor_charge,
                "posterior_error_probability": float("nan"),
                "is_decoy": is_decoy,
                "calculated_mz": float("nan"),
                "observed_mz": observed_mz,
                "additional_scores": np.array([], dtype=object),
                "protein_accessions": np.array([], dtype=object),
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


def _empty_psms_df() -> pd.DataFrame:
    """Return an empty DataFrame with the full psms schema."""
    return pd.DataFrame(columns=_PSM_COLUMNS)
