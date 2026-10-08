"""
Target-modification specification shared by AScore / PhosphoRS / LucXor (issue #57).

A ``PTMSpec`` names the modification being localized, the residues it can sit
on, and the decoy residues used for the decoy-amino-acid FLR. Phospho (S/T/Y,
decoy A) is the default and reproduces the historical behaviour.
"""

import logging
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

from pyopenms import ModificationsDB, ResidueModification

logger = logging.getLogger(__name__)

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
DECOY_SUFFIX = "Decoy"

# Pseudo-UniMod record ids for decoys OpenMS does not ship (it uses 99900-99913).
_DECOY_RECORD_ID_BASE = 990000
_DECOY_RECORD_ID_SPAN = 9000


@dataclass(frozen=True)
class PTMSpec:
    """A modification to localize: pyOpenMS short name, target and decoy residues."""

    name: str
    residues: FrozenSet[str]
    decoy_residues: FrozenSet[str] = frozenset("A")
    neutral_losses: Tuple[Tuple[str, float], ...] = ()

    @property
    def decoy_name(self) -> str:
        return f"{self.name}{DECOY_SUFFIX}"

    @property
    def mass(self) -> float:
        return modification_mass(self.name, min(self.residues))

    @property
    def target_mod_names(self) -> Tuple[str, ...]:
        """Residue-qualified names, e.g. ("Phospho (S)", "Phospho (T)", "Phospho (Y)")."""
        return tuple(f"{self.name} ({aa})" for aa in sorted(self.residues))


PRESETS: Dict[str, PTMSpec] = {
    "phospho": PTMSpec("Phospho", frozenset("STY"), neutral_losses=(("H3PO4", -97.97690),)),
    "acetyl": PTMSpec("Acetyl", frozenset("K")),
    "methyl": PTMSpec("Methyl", frozenset("KR")),
    "dimethyl": PTMSpec("Dimethyl", frozenset("KR")),
    "trimethyl": PTMSpec("Trimethyl", frozenset("K")),
    "gg": PTMSpec("GG", frozenset("K")),
}
_ALIASES = {"glygly": "gg", "ubiquitin": "gg", "ub": "gg"}

PHOSPHO = PRESETS["phospho"]


@lru_cache(maxsize=None)
def modification_mass(name: str, residue: str) -> float:
    """Monoisotopic delta mass of ``name`` on ``residue`` from ModificationsDB."""
    mod = ModificationsDB().getModification(
        name, residue, ResidueModification.TermSpecificity.ANYWHERE
    )
    return mod.getDiffMonoMass()


def get_ptm(
    name: str,
    residues: Optional[Iterable[str]] = None,
    decoy_residues: Optional[Iterable[str]] = None,
) -> PTMSpec:
    """
    Resolve a preset by name (case-insensitive), or build a custom spec.

    ``residues`` / ``decoy_residues`` override the preset when not None and
    accept "STY", "S,T,Y", "S T Y" or a list; a name that is not a preset
    requires ``residues``. The modification must exist in ModificationsDB on
    every target residue, and decoy residues must not overlap the targets.
    """
    key = _ALIASES.get(name.lower(), name.lower())
    base = PRESETS.get(key)
    if base is None:
        if residues is None:
            raise ValueError(
                f"'{name}' is not a preset ({', '.join(sorted(PRESETS))}); "
                "pass the target residues explicitly"
            )
        base = PTMSpec(name, frozenset())

    spec = PTMSpec(
        name=base.name,
        residues=base.residues if residues is None else _parse_residues(residues, "target"),
        decoy_residues=(
            base.decoy_residues if decoy_residues is None else _parse_residues(decoy_residues, "decoy")
        ),
        neutral_losses=base.neutral_losses,
    )
    _validate(spec)
    return spec


def _parse_residues(value: Iterable[str], kind: str) -> FrozenSet[str]:
    """Residue letters from "STY", "S,T,Y", "S T Y" or a list; empty is an error."""
    text = value if isinstance(value, str) else "".join(value)
    parsed = frozenset(re.sub(r"[\s,;]+", "", text).upper())
    if not parsed:
        raise ValueError(f"empty {kind} residues")
    return parsed


def _validate(spec: PTMSpec) -> None:
    bad = (spec.residues | spec.decoy_residues) - set(STANDARD_AAS)
    if bad:
        raise ValueError(f"Not standard amino acids: {sorted(bad)}")
    if not spec.residues:
        raise ValueError(f"{spec.name}: no target residues")
    if not spec.decoy_residues:
        raise ValueError(f"{spec.name}: no decoy residues")
    overlap = spec.residues & spec.decoy_residues
    if overlap:
        raise ValueError(f"{spec.name}: decoy residues overlap targets: {sorted(overlap)}")
    for aa in spec.residues:
        try:
            modification_mass(spec.name, aa)
        except Exception as e:
            raise ValueError(f"'{spec.name}' is not defined on {aa} in ModificationsDB") from e


def is_decoy_mod_name(mod_name: str) -> bool:
    """True for any decoy modification name, with or without a residue suffix."""
    return mod_name.split(" (")[0].endswith(DECOY_SUFFIX)


def decoy_record_id(spec: PTMSpec) -> int:
    """Deterministic pseudo-UniMod record id for a decoy OpenMS does not ship."""
    return _DECOY_RECORD_ID_BASE + zlib.crc32(spec.decoy_name.encode()) % _DECOY_RECORD_ID_SPAN


_REGISTERED: set = set()


def register_decoy(spec: PTMSpec) -> None:
    """
    Register ``<name>Decoy`` on every standard residue at the target mass.

    If OpenMS already ships the decoy (PhosphoDecoy on A/G/L), the remaining
    residues are added under residue-qualified ids, as before. Otherwise the
    short name is defined on every residue with a pseudo-UniMod id so it
    round-trips through idParquet.
    """
    if spec.decoy_name in _REGISTERED:
        return

    from onsite.idparquet import register_modification

    db = ModificationsDB()
    builtin = db.has(spec.decoy_name)
    mass = spec.mass
    record_id = None if builtin else decoy_record_id(spec)

    for aa in STANDARD_AAS:
        full_id = f"{spec.decoy_name} ({aa})"
        try:
            mod = ResidueModification()
            mod.setId(full_id if builtin else spec.decoy_name)
            mod.setFullId(full_id)
            mod.setName(spec.decoy_name)
            mod.setDiffMonoMass(mass)
            mod.setOrigin(aa)
            if record_id is not None:
                mod.setUniModRecordId(record_id)
            db.addModification(mod)
            if record_id is not None:
                register_modification(mod)
        except Exception as e:
            logger.debug(f"{full_id} registration: {e}")

    _REGISTERED.add(spec.decoy_name)
