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
# Modification name mapping: UNIMOD ↔ pyOpenMS short name
# ---------------------------------------------------------------------------
# The pyOpenMS AASequence uses short modification names (e.g. "Phospho"),
# while idParquet uses ProForma UNIMOD notation (e.g. "[UNIMOD:21]").
#
# We build the mapping from a hardcoded fallback for the most common
# modifications, supplemented by ModificationsDB if pyOpenMS is available.
# ---------------------------------------------------------------------------

_UNIMOD_TO_PYO: Dict[str, str] = {}   # "unimod:21" -> "Phospho"
_PYO_TO_UNIMOD: Dict[str, str] = {}   # "Phospho"    -> "UNIMOD:21"

def _ensure_mod_mappings():
    """Populate UNIMOD ↔ pyOpenMS name caches from ModificationsDB."""
    if _UNIMOD_TO_PYO:
        return
    db = ModificationsDB()
    n = db.getNumberOfModifications()
    for i in range(n):
        mod = db.getModification(i)
        uname = mod.getUniModAccession()
        sname = mod.getName()
        fname = mod.getFullName()
        idname = mod.getId()
        parts = uname.split(":")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        unimod_str = f"UNIMOD:{parts[1]}"

        # getId() returns the short name AASequence.toString() uses (e.g. "Phospho")
        for name in (sname, fname, idname):
            if name and name not in _PYO_TO_UNIMOD:
                _PYO_TO_UNIMOD[name] = unimod_str

        uname_lower = uname.lower()
        short_name = sname or idname or fname or unimod_str
        if uname_lower not in _UNIMOD_TO_PYO:
            _UNIMOD_TO_PYO[uname_lower] = short_name


# ---------------------------------------------------------------------------
# String-level conversion: UNIMOD notation ↔ pyOpenMS notation
# These are pure string functions (no pyOpenMS dependency).
# ---------------------------------------------------------------------------


def unimod_to_pyopenms_notation(peptidoform: str) -> str:
    """Convert ProForma UNIMOD notation (``S[UNIMOD:21]``) to pyOpenMS notation (``S(Phospho)``)."""
    _ensure_mod_mappings()

    def _lookup(num: str) -> str:
        name = _UNIMOD_TO_PYO.get(f"unimod:{num}")
        return name if name else num

    peptidoform = re.sub(
        r"\[UNIMOD:(\d+)\]-",
        lambda m: f"({_lookup(m.group(1))})",
        peptidoform,
    )
    peptidoform = re.sub(
        r"-\[UNIMOD:(\d+)\]",
        lambda m: f"({_lookup(m.group(1))})",
        peptidoform,
    )
    peptidoform = re.sub(
        r"([A-Z])\[UNIMOD:(\d+)\]",
        lambda m: f"{m.group(1)}({_lookup(m.group(2))})",
        peptidoform,
    )
    return peptidoform


def pyopenms_to_unimod_notation(seq_str: str) -> str:
    """Convert pyOpenMS notation (``S(Phospho)``) back to ProForma UNIMOD notation (``S[UNIMOD:21]``)."""
    _ensure_mod_mappings()

    def _to_unimod(mod_name: str) -> str:
        return _PYO_TO_UNIMOD.get(mod_name, mod_name)

    seq_str = re.sub(
        r"^\.\(([^)]+)\)",
        lambda m: f"[{_to_unimod(m.group(1))}]-",
        seq_str,
    )
    # N-terminal without leading dot: "(name)seq" -> "[UNIMOD:N]-seq"
    seq_str = re.sub(
        r"^\(([^)]+)\)([A-Z])",
        lambda m: f"[{_to_unimod(m.group(1))}]-{m.group(2)}",
        seq_str,
    )
    seq_str = re.sub(
        r"([A-Z])\(([^)]+)\)",
        lambda m: f"{m.group(1)}[{_to_unimod(m.group(2))}]",
        seq_str,
    )
    return seq_str


def peptidoform_to_modifications(peptidoform: str, site_scores: Optional[Dict[int, float]] = None):
    """Parse a ProForma peptidoform into the ``modifications`` column format"""
    _ensure_mod_mappings()
    if not peptidoform or "[UNIMOD:" not in peptidoform:
        return np.array([], dtype=object)
    mods = []
    nterm = re.match(r"^\[UNIMOD:(\d+)\]-", peptidoform)
    if nterm:
        uname = f"UNIMOD:{nterm.group(1)}"
        name = _UNIMOD_TO_PYO.get(f"unimod:{nterm.group(1)}", uname)
        mods.append({
            "name": name, "accession": uname,
            "positions": np.array([{"position": "N-term.0", "scores": None}], dtype=object),
        })
        peptidoform = peptidoform[nterm.end():]
    matches = list(re.finditer(r"([A-Z])\[UNIMOD:(\d+)\]", peptidoform))
    mod_positions = defaultdict(list)
    for m in matches:
        residue = m.group(1)
        key = f"UNIMOD:{m.group(2)}"
        pos_in_peptido = m.start()
        idx = 0
        pos = 0
        while pos < len(peptidoform) and pos < pos_in_peptido:
            if peptidoform[pos] == "[":
                j = peptidoform.find("]", pos)
                if j >= 0:
                    pos = j + 1
                    continue
            if peptidoform[pos].isalpha():
                idx += 1
            pos += 1
        score = None
        if site_scores and (idx + 1) in site_scores:
            score = float(site_scores[idx + 1])
        mod_positions[key].append({"position": f"{residue}.{idx + 1}", "scores": score})
    for accession, positions in mod_positions.items():
        unimod_num = accession.split(":")[1]
        name = _UNIMOD_TO_PYO.get(f"unimod:{unimod_num}", accession)
        mods.append({"name": name, "accession": accession, "positions": np.array(positions, dtype=object)})
    return np.array(mods, dtype=object)

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
    if psms_df is None or len(psms_df) == 0:
        print("Warning: No PSMs to save")
        return

    src_psm = os.path.join(source_idparquet, "psms.parquet") if source_idparquet else None
    if src_psm and os.path.isfile(src_psm):
        shutil.copy2(src_psm, os.path.join(out_dir, "psms.parquet"))
    tbl = pq.read_table(os.path.join(out_dir, "psms.parquet"))
    src_cols = set(tbl.schema.names)

    def _pa_col(name, values):
        return pa.array(values, type=tbl.schema.field(name).type)

    updates = {}
    for col in ("peptidoform", "sequence", "score", "score_type", "higher_score_better"):
        if col in psms_df.columns and col in src_cols:
            updates[col] = _pa_col(col, psms_df[col].tolist())
    if "psm_metavalues" in psms_df.columns and "psm_metavalues" in src_cols:
        updates["psm_metavalues"] = _pa_col("psm_metavalues", psms_df["psm_metavalues"].tolist())

    if "peptidoform" in src_cols and "modifications" in src_cols:
        _pf_col = psms_df["peptidoform"] if "peptidoform" in psms_df.columns else None
        _mv_col = psms_df["psm_metavalues"] if "psm_metavalues" in psms_df.columns else None
        if _pf_col is not None:
            _score_meta_keys = ["AScore_site_scores", "Luciphor_site_scores", "PhosphoRS_site_delta"]
            _new_mods = []
            for i in range(len(psms_df)):
                pf = str(_pf_col.iloc[i]) if _pf_col is not None else ""
                ss = None
                if _mv_col is not None:
                    mv = _mv_col.iloc[i]
                    if isinstance(mv, np.ndarray):
                        for m in mv:
                            if isinstance(m, dict) and m.get("name") in _score_meta_keys:
                                try:
                                    d = ast.literal_eval(m["value"])
                                    ss = {int(k): float(v) for k, v in d.items()}
                                except Exception:
                                    pass
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
    """Write a non-PSM parquet file (search_params / protein_groups)"""
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
