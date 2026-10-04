"""
featurize.py

Computes a set of physicochemical molecular descriptors from SMILES strings
using RDKit. These descriptors are the input features for the solubility
prediction model.

The descriptor set is based on Delaney's original ESOL paper (2004), which
showed that aqueous solubility correlates strongly with a small number of
simple 2D descriptors: molecular weight, lipophilicity (logP), polarity
(TPSA, H-bond donors/acceptors), flexibility (rotatable bonds), and
aromaticity (ring count).

Reference:
    Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility Directly
    from Molecular Structure. J. Chem. Inf. Comput. Sci., 44(3), 1000-1005.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)

# Names of the descriptor columns produced by `compute_descriptors`.
DESCRIPTOR_COLUMNS = [
    "MolWt",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "AromaticProportion",
]


def _aromatic_proportion(mol: Chem.Mol) -> float:
    """Fraction of heavy atoms that are aromatic.

    Highly aromatic molecules tend to be flatter and pack more tightly in
    the solid state, which generally lowers aqueous solubility.
    """
    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy == 0:
        return 0.0
    num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    return num_aromatic / num_heavy


def compute_descriptors(smiles: object) -> dict | None:
    """Compute RDKit descriptors for a single SMILES string.

    Args:
        smiles: A SMILES string, e.g. "CCO". Also accepts non-string input
            (e.g. NaN/None from a CSV with missing values) and treats it
            as invalid rather than crashing.

    Returns:
        A dict mapping descriptor name -> value, or None if the input was
        missing, empty, or could not be parsed by RDKit.

    Note:
        RDKit silently parses an empty string ("") as a valid zero-atom
        molecule and returns all-zero descriptors for it, rather than
        failing. Left unchecked, this would inject a fake all-zero
        "molecule" into the dataset for any row with a missing SMILES
        value. We reject empty/whitespace-only and non-string input
        explicitly, before it ever reaches RDKit.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        logger.warning("Skipping missing/empty SMILES: %r", smiles)
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Could not parse SMILES: %r", smiles)
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "AromaticProportion": _aromatic_proportion(mol),
    }


def featurize_dataframe(
    df: pd.DataFrame, smiles_col: str = "smiles"
) -> pd.DataFrame:
    """Add descriptor columns to a DataFrame that has a SMILES column.

    Rows whose SMILES cannot be parsed are dropped (with a warning logged
    for each one), since they cannot be featurized.

    Args:
        df: Input DataFrame containing a column of SMILES strings.
        smiles_col: Name of the column holding SMILES strings.

    Returns:
        A new DataFrame with the original columns plus one column per
        entry in DESCRIPTOR_COLUMNS. Unparseable rows are removed.
    """
    records = []
    kept_idx = []
    for idx, smiles in df[smiles_col].items():
        desc = compute_descriptors(smiles)
        if desc is not None:
            records.append(desc)
            kept_idx.append(idx)

    n_dropped = len(df) - len(kept_idx)
    if n_dropped:
        logger.warning("Dropped %d row(s) with unparseable SMILES", n_dropped)

    # If every row failed to parse (or df was empty), `records` is an
    # empty list and `pd.DataFrame(records, ...)` would produce a frame
    # with NO columns at all -- silently dropping DESCRIPTOR_COLUMNS from
    # the output instead of returning an (empty) frame with the expected
    # shape. That breaks any downstream code indexing by column name
    # (e.g. `df[DESCRIPTOR_COLUMNS]` in train.py would raise a KeyError
    # instead of a clear "no usable data" signal). We force the expected
    # columns explicitly so the empty case fails predictably.
    desc_df = pd.DataFrame(records, index=kept_idx, columns=DESCRIPTOR_COLUMNS)
    return df.loc[kept_idx].join(desc_df)
