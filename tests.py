"""
tests.py

All tests for this project, in one file, run with: pytest tests.py -v

Two groups of tests:
  1. Original unit tests for src/featurize.py (valid SMILES, descriptor
     values, dataframe featurization).
  2. Regression tests for three bugs found during an explicit error-check
     pass (see README "Error check & validation" section for details):
       - empty string silently treated as a valid zero-atom molecule
       - None / NaN input crashing instead of being skipped
       - an all-invalid input dataframe losing its descriptor columns
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from featurize import DESCRIPTOR_COLUMNS, compute_descriptors, featurize_dataframe

# ---------------------------------------------------------------------------
# Original tests: basic correctness of descriptor computation
# ---------------------------------------------------------------------------


def test_compute_descriptors_valid_smiles():
    """Ethanol should produce the expected molecular weight and the
    correct, known H-bond donor count."""
    result = compute_descriptors("CCO")
    assert result is not None
    assert set(result.keys()) == set(DESCRIPTOR_COLUMNS)
    assert result["MolWt"] == pytest.approx(46.07, abs=0.05)
    assert result["NumHDonors"] == 1


def test_compute_descriptors_invalid_smiles_returns_none():
    """Garbage input should not raise — it should return None so the
    caller can decide how to handle it."""
    assert compute_descriptors("this is not a smiles string") is None


def test_aromatic_proportion_benzene_is_one():
    """Benzene is fully aromatic: all 6 heavy atoms are aromatic carbons."""
    result = compute_descriptors("c1ccccc1")
    assert result["AromaticProportion"] == pytest.approx(1.0)


def test_aromatic_proportion_ethanol_is_zero():
    """Ethanol has no aromatic atoms."""
    result = compute_descriptors("CCO")
    assert result["AromaticProportion"] == pytest.approx(0.0)


def test_featurize_dataframe_drops_unparseable_rows():
    df = pd.DataFrame({"smiles": ["CCO", "not a smiles", "c1ccccc1"]})
    result = featurize_dataframe(df, smiles_col="smiles")
    assert len(result) == 2
    assert set(result["smiles"]) == {"CCO", "c1ccccc1"}
    for col in DESCRIPTOR_COLUMNS:
        assert col in result.columns


# ---------------------------------------------------------------------------
# Regression tests for bugs found during the error-check pass
# ---------------------------------------------------------------------------


def test_empty_string_is_rejected_not_treated_as_zero_molecule():
    """BUG (fixed): RDKit parses "" as a valid zero-atom molecule and
    returns all-zero descriptors for it. Left unchecked, a missing SMILES
    stored as "" in a CSV would silently become a fake all-zero data
    point in the training set instead of being dropped. compute_descriptors
    must reject it explicitly.
    """
    assert compute_descriptors("") is None


def test_whitespace_only_string_is_rejected():
    assert compute_descriptors("   ") is None


def test_none_input_does_not_raise():
    """BUG (fixed): passing None (e.g. from a CSV cell read as NaN then
    cast) used to raise an unhandled TypeError from RDKit's C++ bindings,
    crashing the whole featurization loop on one bad row. It must now
    return None instead of raising.
    """
    assert compute_descriptors(None) is None


def test_nan_float_input_does_not_raise():
    """BUG (fixed): a missing value read by pandas from a CSV becomes
    float('nan'), not None or "". This used to raise the same unhandled
    TypeError as the None case above.
    """
    assert compute_descriptors(float("nan")) is None


def test_featurize_dataframe_all_invalid_keeps_descriptor_columns():
    """BUG (fixed): when every row fails to parse, `records` is an empty
    list, and `pd.DataFrame(records)` with no `columns=` argument produces
    a frame with ZERO columns -- silently dropping DESCRIPTOR_COLUMNS
    instead of returning an empty-but-correctly-shaped frame. Any
    downstream code selecting `df[DESCRIPTOR_COLUMNS]` (as train.py does)
    would then raise a confusing KeyError instead of a clear "no usable
    data" signal. The fix passes `columns=DESCRIPTOR_COLUMNS` explicitly.
    """
    df = pd.DataFrame({"smiles": ["garbage1", "garbage2", "not_a_smiles"]})
    result = featurize_dataframe(df, smiles_col="smiles")
    assert len(result) == 0
    for col in DESCRIPTOR_COLUMNS:
        assert col in result.columns


def test_featurize_dataframe_empty_input_keeps_descriptor_columns():
    """Same bug class as above, triggered by an empty input dataframe
    rather than an all-invalid one."""
    df = pd.DataFrame({"smiles": []})
    result = featurize_dataframe(df, smiles_col="smiles")
    assert len(result) == 0
    for col in DESCRIPTOR_COLUMNS:
        assert col in result.columns


def test_featurize_dataframe_mix_of_missing_and_valid_smiles():
    """Integration-style check: a realistic messy CSV column containing a
    valid SMILES, an empty string, and a NaN, all in one pass -- this is
    the exact combination that triggered the empty-string and NaN bugs
    together before the fix.
    """
    df = pd.DataFrame({"smiles": ["CCO", "", float("nan"), "c1ccccc1"]})
    result = featurize_dataframe(df, smiles_col="smiles")
    assert len(result) == 2
    assert set(result["smiles"]) == {"CCO", "c1ccccc1"}
