#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""idParquet I/O module for the OnSite package."""

import ast
import re
import time
import logging
import os
from collections import defaultdict
from typing import Dict, Optional, Tuple
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyopenms import ModificationsDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ModMapper — UNIMOD ↔ pyOpenMS name mapping + peptidoform parsing
# ---------------------------------------------------------------------------
# The pyOpenMS AASequence uses short modification names (e.g. "Phospho"),
# while idParquet uses ProForma UNIMOD notation (e.g. "[UNIMOD:21]").
#
# ModMapper lazily loads the mapping from ModificationsDB on first use,
# then exposes converters and the peptidoform-to-modifications parser.
# All state lives on the singleton ``_MAPPER`` instance; module-level
# functions below it are thin wrappers for backward compatibility.
# ---------------------------------------------------------------------------


class ModMapper:
    """
    UNIMOD ↔ pyOpenMS modification name mapper and peptidoform parser.
    
    convert UNIMOD accessions in ProForma peptidoforms to pyOpenMS names for AASequence parsing
    and convert pyOpenMS-style strings back to UNIMOD notation for idParquet output.
    """

    def __init__(self):
        """Initialize empty mappers; lazily populated from ModificationsDB on first use."""
        self._unimod_to_pyo: Dict[str, str] = {}  # "unimod:21" -> "Phospho"
        self._pyo_to_unimod: Dict[str, str] = {}  # "Phospho"    -> "UNIMOD:21"

    # -- lazy loading -------------------------------------------------------

    def _ensure(self):
        if not self._unimod_to_pyo:
            self._load_from_db()

    def _load_from_db(self):
        db = ModificationsDB()
        for i in range(db.getNumberOfModifications()):
            self._register_mod(db.getModification(i))

    def _register_mod(self, mod):
        uname = mod.getUniModAccession()
        parts = uname.split(":")
        if len(parts) != 2 or not parts[1].isdigit():
            return
        u = f"UNIMOD:{parts[1]}"
        self._populate_pyo_to_unimod(mod, u)
        self._populate_unimod_to_pyo(uname, mod, u)

    def _populate_pyo_to_unimod(self, mod, unimod_str):
        for name in (mod.getName(), mod.getFullName(), mod.getId()):
            if name and name not in self._pyo_to_unimod:
                self._pyo_to_unimod[name] = unimod_str

    def _populate_unimod_to_pyo(self, uname, mod, unimod_str):
        short = next(n for n in (mod.getName(), mod.getId(), mod.getFullName(), unimod_str) if n)
        key = uname.lower()
        if key not in self._unimod_to_pyo:
            self._unimod_to_pyo[key] = short

    # -- public converters --------------------------------------------------

    def unimod_to_pyopenms(self, peptidoform: str) -> str:
        self._ensure()

        def _lookup(num: str) -> str:
            return self._unimod_to_pyo.get(f"unimod:{num}", num)

        s = re.sub(r"\[UNIMOD:(\d+)\]-", lambda m: f"({_lookup(m.group(1))})", peptidoform)
        s = re.sub(r"-\[UNIMOD:(\d+)\]", lambda m: f"({_lookup(m.group(1))})", s)
        s = re.sub(r"([A-Z])\[UNIMOD:(\d+)\]", lambda m: f"{m.group(1)}({_lookup(m.group(2))})", s)
        return s

    def pyopenms_to_unimod(self, seq_str: str) -> str:
        self._ensure()

        def _to_u(name: str) -> str:
            return self._pyo_to_unimod.get(name, name)

        s = re.sub(r"^\.\(([^)]+)\)", lambda m: f"[{_to_u(m.group(1))}]-", seq_str)
        s = re.sub(r"^\(([^)]+)\)([A-Z])", lambda m: f"[{_to_u(m.group(1))}]-{m.group(2)}", s)
        s = re.sub(r"([A-Z])\(([^)]+)\)", lambda m: f"{m.group(1)}[{_to_u(m.group(2))}]", s)
        return s

    # -- modifications parser -----------------------------------------------

    def peptidoform_to_modifications(self, peptidoform: str,
                                    site_scores: Optional[Dict[int, float]] = None):
        """Parse a ProForma peptidoform into the ``modifications`` column format."""
        self._ensure()
        if not peptidoform or "[UNIMOD:" not in peptidoform:
            return np.array([], dtype=object)

        mods = []

        # N-terminal
        nterm = re.match(r"^\[UNIMOD:(\d+)\]-", peptidoform)
        if nterm:
            acc = f"UNIMOD:{nterm.group(1)}"
            name = self._unimod_to_pyo.get(f"unimod:{nterm.group(1)}", acc)
            mods.append({
                "name": name, "accession": acc,
                "positions": np.array([{"position": "N-term.0", "scores": None}], dtype=object),
            })
            peptidoform = peptidoform[nterm.end():]

        # Residue-specific
        for acc, positions in self._group_residue_mods(peptidoform, site_scores).items():
            num = acc.split(":")[1]
            name = self._unimod_to_pyo.get(f"unimod:{num}", acc)
            mods.append({"name": name, "accession": acc,
                        "positions": np.array(positions, dtype=object)})

        return np.array(mods, dtype=object)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _compute_residue_index_map(peptidoform: str):
        """Return a list where list[pos] = 1-based residue index (0 = not a residue)."""
        res_idx = [0] * len(peptidoform)
        depth = cnt = 0
        for i, c in enumerate(peptidoform):
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
            elif c.isalpha() and depth == 0:
                cnt += 1
                res_idx[i] = cnt
        return res_idx

    def _group_residue_mods(self, peptidoform: str, site_scores):
        """Group residue-position data by modification accession."""
        res_idx = self._compute_residue_index_map(peptidoform)
        groups = defaultdict(list)
        for m in re.finditer(r"([A-Z])\[UNIMOD:(\d+)\]", peptidoform):
            residue, acc_num = m.group(1), m.group(2)
            key = f"UNIMOD:{acc_num}"
            r_idx = res_idx[m.start()]
            score = site_scores.get(r_idx) if site_scores else None
            groups[key].append({"position": f"{residue}.{r_idx}", "scores": score})
        return groups


# -- Singleton instance -------------------------------------------------------

_MAPPER = ModMapper()


# -- Module-level convenience wrappers (backward-compatible API) --------------

def unimod_to_pyopenms_notation(peptidoform: str) -> str:
    return _MAPPER.unimod_to_pyopenms(peptidoform)


def pyopenms_to_unimod_notation(seq_str: str) -> str:
    return _MAPPER.pyopenms_to_unimod(seq_str)


def peptidoform_to_modifications(peptidoform: str, site_scores: Optional[Dict[int, float]] = None):
    return _MAPPER.peptidoform_to_modifications(peptidoform, site_scores)

# ---------------------------------------------------------------------------
# Path resolution helper
# ---------------------------------------------------------------------------


def resolve_parquet_path(idparquet_path: str) -> Tuple[str, str]:
    """Resolve an idparquet path and return (resolved_dir, psm_parquet_path)."""
    psm_path = os.path.join(idparquet_path, "psms.parquet")
    if not os.path.isdir(idparquet_path) or not os.path.isfile(psm_path):
        alt = idparquet_path.rstrip("/\\") + ".idparquet"
        if os.path.isdir(alt):
            idparquet_path = alt
            psm_path = os.path.join(alt, "psms.parquet")
    return idparquet_path, psm_path


# ---------------------------------------------------------------------------
# Main load / save functions (DataFrame-based, no pyOpenMS)
# ---------------------------------------------------------------------------


def load_dataframes(
    idparquet_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load identification data from an idParquet directory as DataFrames."""
    idparquet_path, psm_path = resolve_parquet_path(idparquet_path)
    print(
        f"[{time.strftime('%H:%M:%S')}] Loading identifications from {idparquet_path}"
    )

    psms_df = pd.read_parquet(psm_path)
    print(f"Loaded {len(psms_df)} PSMs")

    prot_path = os.path.join(idparquet_path, "proteins.parquet")
    if os.path.isfile(prot_path):
        proteins_df = pd.read_parquet(prot_path)
        print(f"Loaded {len(proteins_df)} proteins")
    else:
        proteins_df = pd.DataFrame()

    sp_path = os.path.join(idparquet_path, "search_params.parquet")
    if os.path.isfile(sp_path):
        search_params_df = pd.read_parquet(sp_path)
    else:
        search_params_df = pd.DataFrame()

    pg_path = os.path.join(idparquet_path, "protein_groups.parquet")
    if os.path.isfile(pg_path):
        protein_groups_df = pd.read_parquet(pg_path)
    else:
        protein_groups_df = pd.DataFrame()

    return psms_df, proteins_df, search_params_df, protein_groups_df


def save_dataframes(
    out_path: str,
    psms_df: pd.DataFrame,
    proteins_df: Optional[pd.DataFrame] = None,
    run_identifier: Optional[str] = None,
    template_df: Optional[pd.DataFrame] = None,
    source_idparquet: Optional[str] = None,
) -> str:
    """Save identification DataFrames to an idParquet directory."""
    out_dir = out_path if out_path.endswith(".idparquet") else out_path + ".idparquet"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] Saving results to {out_dir}")

    _save_psms_parquet(out_dir, psms_df, source_idparquet)
    _save_proteins_parquet(out_dir, proteins_df, source_idparquet, run_identifier)
    _copy_or_create_parquet("search_params", out_dir, source_idparquet, run_identifier)
    _copy_or_create_parquet("protein_groups", out_dir, source_idparquet, run_identifier)

    print(f"Results saved to {out_dir}")
    return out_dir


def _save_psms_parquet(out_dir: str, psms_df: pd.DataFrame, source_idparquet: Optional[str]):
    """Write psms.parquet by updating columns of the source schema."""
    src_psm = os.path.join(source_idparquet, "psms.parquet")
    shutil.copy2(src_psm, os.path.join(out_dir, "psms.parquet"))
    tbl = pq.read_table(os.path.join(out_dir, "psms.parquet"))

    def _pa_col(name, values):
        return pa.array(values, type=tbl.schema.field(name).type)

    updates = {}
    for col in ("peptidoform",):
        updates[col] = _pa_col(col, psms_df[col].tolist())

    updates["psm_metavalues"] = _pa_col("psm_metavalues", psms_df["psm_metavalues"].tolist())

    _pf_col = psms_df["peptidoform"] 
    _mv_col = psms_df["psm_metavalues"]
    _score_meta_keys = ["AScore_site_scores", "Luciphor_site_scores", "PhosphoRS_site_delta"]
    _new_mods = []
    for i in range(len(psms_df)):
        pf = str(_pf_col.iloc[i])
        ss = None
        mv = _mv_col.iloc[i]
        for m in mv:
            if m.get("name") in _score_meta_keys:
                d = ast.literal_eval(m["value"])
                ss = {int(k): float(v) for k, v in d.items()}
                break
        _new_mods.append(peptidoform_to_modifications(pf, ss))
    updates["modifications"] = _pa_col("modifications", _new_mods)

    for name, arr in updates.items():
        idx = tbl.schema.get_field_index(name)
        if idx >= 0:
            tbl = tbl.set_column(idx, tbl.schema.field(idx), arr)
    pq.write_table(tbl, os.path.join(out_dir, "psms.parquet"))
    print(f"Saved {len(psms_df)} PSMs to psms.parquet")


def _save_proteins_parquet(out_dir: str, proteins_df: Optional[pd.DataFrame],
                            source_idparquet: Optional[str], run_identifier: Optional[str]):
    """Write proteins.parquet — copy from source or create from DataFrame."""
    src = os.path.join(source_idparquet, "proteins.parquet") if source_idparquet else None
    if src and os.path.isfile(src):
        shutil.copy2(src, os.path.join(out_dir, "proteins.parquet"))
    elif proteins_df is not None and len(proteins_df) > 0:
        df = proteins_df.copy()
        if "run_identifier" not in df.columns:
            df["run_identifier"] = run_identifier or ""
        df.to_parquet(os.path.join(out_dir, "proteins.parquet"), index=False)
        print(f"Saved {len(df)} proteins to proteins.parquet")


def _copy_or_create_parquet(name: str, out_dir: str,
                            source_idparquet: Optional[str],
                            run_identifier: Optional[str]):
    """Write a non-PSM parquet file (search_params / protein_groups)."""
    path = os.path.join(out_dir, f"{name}.parquet")
    src = os.path.join(source_idparquet, f"{name}.parquet") if source_idparquet else None
    if src and os.path.isfile(src):
        shutil.copy2(src, path)
        return
    if os.path.isfile(path):
        return

    if name == "search_params":
        row = {
            "run_identifier": run_identifier or "",
            "search_engine": "", "search_engine_version": "",
            "score_type": "", "higher_score_better": True,
            "significance_threshold": 0.0, "db": "", "charges": "",
            "mass_type": "MONOISOTOPIC", "digestion_enzyme": "",
            "missed_cleavages": 0,
            "variable_modifications": np.array([], dtype=object),
            "fixed_modifications": np.array([], dtype=object),
        }
    elif name == "protein_groups":
        row = {
            "group_type": "", "probability": 0.0,
            "accessions": np.array([], dtype=object),
            "run_identifier": run_identifier or "",
        }
    else:
        return
    pd.DataFrame([row]).to_parquet(path, index=False)


def save_psms_from_scratch(
    out_dir: str,
    psms_df: pd.DataFrame,
    proteins_df: Optional[pd.DataFrame] = None,
) -> str:
    """Write an idParquet directory from a psms DataFrame without requiring a source.

    Parameters
    ----------
    out_dir : str
        Output directory path. Will be created if it does not exist.
        If it does not end with ``.idparquet``, the suffix is appended.
    psms_df : pd.DataFrame
        PSMs DataFrame — must be schema-compatible with the 29-column psms schema.
    proteins_df : pd.DataFrame, optional
        Proteins DataFrame. If None, an empty proteins.parquet is written.

    Returns
    -------
    str
        Absolute path to the created idParquet directory.
    """
    if not out_dir.endswith(".idparquet"):
        out_dir = out_dir + ".idparquet"
    os.makedirs(out_dir, exist_ok=True)
    logger.info("Saving PSMs from scratch to %s", out_dir)

    # Import column list from id_io to avoid duplication
    from onsite.id_io import _PSM_COLUMNS, _PSM_DTYPE_MAP

    # Ensure all required columns exist with correct defaults
    df = psms_df.copy()
    _int32_cols = {"precursor_charge", "scan", "hit_index", "peptide_identification_index"}
    _float64_cols = {
        "score", "rt", "observed_mz", "calculated_mz",
        "posterior_error_probability", "predicted_rt", "ion_mobility",
    }
    _bool_cols = {"is_decoy", "higher_score_better"}

    for col in _PSM_COLUMNS:
        if col not in df.columns:
            if col in _int32_cols:
                df[col] = np.int32(0)
            elif col in _float64_cols:
                df[col] = float("nan")
            elif col in _bool_cols:
                df[col] = False
            else:
                df[col] = None

    # Enforce dtypes
    for col, dtype in _PSM_DTYPE_MAP.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    # Reorder columns to match schema
    df = df[[c for c in _PSM_COLUMNS if c in df.columns]]

    # Write psms.parquet
    df.to_parquet(os.path.join(out_dir, "psms.parquet"), index=False)
    logger.info("Saved %d PSMs to psms.parquet", len(df))

    # Write proteins.parquet
    if proteins_df is not None and len(proteins_df) > 0:
        proteins_df.to_parquet(os.path.join(out_dir, "proteins.parquet"), index=False)
        logger.info("Saved %d proteins to proteins.parquet", len(proteins_df))
    else:
        pd.DataFrame().to_parquet(os.path.join(out_dir, "proteins.parquet"), index=False)

    # Write search_params.parquet and protein_groups.parquet (minimal rows)
    _copy_or_create_parquet("search_params", out_dir, None, None)
    _copy_or_create_parquet("protein_groups", out_dir, None, None)

    logger.info("Saved idParquet to %s", out_dir)
    return out_dir
