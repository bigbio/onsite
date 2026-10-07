"""
Tests for the target-modification spec (onsite/ptm.py, #57).
"""

import pytest
from pyopenms import AASequence

from onsite import ptm
from onsite.decoy_flr import PSMRecord, compute_tool_flr, parse_localized_sites
from onsite.idparquet import pyopenms_to_unimod_notation, unimod_to_pyopenms_notation


def test_phospho_preset_matches_historical_constants():
    p = ptm.get_ptm("phospho")
    assert p is not None and p == ptm.PHOSPHO
    assert p.residues == frozenset("STY")
    assert p.decoy_residues == frozenset("A")
    assert p.decoy_name == "PhosphoDecoy"
    assert p.target_mod_names == ("Phospho (S)", "Phospho (T)", "Phospho (Y)")
    assert p.mass == pytest.approx(79.966331, abs=1e-6)


@pytest.mark.parametrize(
    "name,mass",
    [("Acetyl", 42.010565), ("Methyl", 14.01565), ("Dimethyl", 28.0313),
     ("Trimethyl", 42.04695), ("GG", 114.042927), ("GlyGly", 114.042927)],
)
def test_presets_resolve_masses_from_modificationsdb(name, mass):
    assert ptm.get_ptm(name).mass == pytest.approx(mass, abs=1e-4)


def test_overrides_and_custom_modification():
    p = ptm.get_ptm("Phospho", residues="sty", decoy_residues="G")
    assert p.residues == frozenset("STY") and p.decoy_residues == frozenset("G")
    c = ptm.get_ptm("Crotonyl", residues="K")
    assert c.name == "Crotonyl" and c.residues == frozenset("K")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "NotAMod", "residues": "K"},
        {"name": "Crotonyl"},
        {"name": "Acetyl", "residues": "K", "decoy_residues": "K"},
        {"name": "Acetyl", "residues": "KB"},
    ],
)
def test_invalid_specs_raise(kwargs):
    with pytest.raises(ValueError):
        ptm.get_ptm(**kwargs)


def test_is_decoy_mod_name():
    assert ptm.is_decoy_mod_name("PhosphoDecoy")
    assert ptm.is_decoy_mod_name("PhosphoDecoy (S)")
    assert ptm.is_decoy_mod_name("AcetylDecoy (A)")
    assert not ptm.is_decoy_mod_name("Phospho (S)")
    assert not ptm.is_decoy_mod_name("Acetyl")


def test_decoy_record_ids_are_distinct_and_outside_openms_range():
    ids = {ptm.decoy_record_id(p) for k, p in ptm.PRESETS.items() if k != "phospho"}
    assert len(ids) == len(ptm.PRESETS) - 1
    assert all(i >= 990000 for i in ids)


def test_registered_decoy_parses_and_round_trips_through_unimod():
    acetyl = ptm.get_ptm("Acetyl")
    # Load the mapper before registering to exercise the late-registration path.
    unimod_to_pyopenms_notation("PEPK[UNIMOD:1]R")
    ptm.register_decoy(acetyl)

    seq = AASequence.fromString("PEPA(AcetylDecoy)K(Acetyl)R")
    unmod = AASequence.fromString("PEPAKR").getMonoWeight()
    assert seq.getMonoWeight() == pytest.approx(unmod + 2 * acetyl.mass, abs=1e-6)

    u = pyopenms_to_unimod_notation(seq.toString())
    assert f"A[UNIMOD:{ptm.decoy_record_id(acetyl)}]" in u
    assert unimod_to_pyopenms_notation(u) == "PEPA(AcetylDecoy)K(Acetyl)R"


def test_parse_localized_sites_respects_spec():
    acetyl = ptm.get_ptm("Acetyl")
    seq = "PEPA(AcetylDecoy)K(Acetyl)S(Phospho)R"
    _, sites = parse_localized_sites(seq, acetyl)
    assert sites == [(4, "A", "AcetylDecoy"), (5, "K", "Acetyl")]
    _, sites = parse_localized_sites(seq)
    assert sites == [(6, "S", "Phospho")]


def test_compute_tool_flr_counts_spec_residues():
    acetyl = ptm.get_ptm("Acetyl")
    recs = [
        PSMRecord("s1", "PEKAKR", [(3, "K", "Acetyl")], {3: 9.0, 5: 1.0}, False, 0.001),
        PSMRecord("s2", "PEKAKR", [(4, "A", "AcetylDecoy")], {3: 1.0, 4: 5.0}, False, 0.001),
    ]
    keep = {("s1", "PEKAKR"), ("s2", "PEKAKR")}
    res = compute_tool_flr(recs, "ascore", keep, 0.01, 0.05, ptm=acetyl)
    assert res.n_analyzed_psms == 2
    assert (res.t_c, res.x_c) == (4, 2)  # two K and one A per peptide
    assert [d for (_s, d, _c, _k) in res.site_records] == [False, True]
