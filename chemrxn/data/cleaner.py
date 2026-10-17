"""
Chemical reaction cleaner, validator, and canonicalizer using RDKit.
Supports configurable stereochemistry, atom-mapping stripping, and Murcko scaffolds.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemrxn.data.schemas import ProcessedReaction


class ReactionCleaner:
    """
    RDKit-based reaction cleaning and validation engine.
    Strictly preserves chemical semantics, stereochemistry, and separates reagents/conditions.
    """

    def __init__(self, keep_stereochemistry: bool = True, require_single_product: bool = True):
        self.keep_stereochemistry = keep_stereochemistry
        self.require_single_product = require_single_product

    def clean_molecule(self, smiles: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parses, strips atom-mapping, optionally handles stereochemistry, and canonicalizes.
        Returns:
            (canonical_smiles, error_reason)
        """
        if not smiles or not isinstance(smiles, str):
            return None, "empty_molecule"

        s = smiles.strip()
        if not s:
            return None, "empty_molecule"

        # Strip CXSMILES annotations e.g. |f:2.3|
        if "|" in s:
            s = s.split("|")[0].strip()

        try:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return None, "rdkit_parse_failed"

            # Strip atom-mapping while preserving stereochemistry and charges
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    atom.SetAtomMapNum(0)

            # Optional stereochemistry removal if explicitly configured
            if not self.keep_stereochemistry:
                Chem.RemoveStereochemistry(mol)

            # Sanitize to check kekulization and valences
            Chem.SanitizeMol(mol)

            canonical = Chem.MolToSmiles(mol, canonical=True)
            if not canonical:
                return None, "empty_canonical_smiles"

            return canonical, None
        except Exception as e:
            return None, f"unsupported_structure: {str(e)[:40]}"

    def extract_scaffold(self, mol_smiles: str) -> str:
        """
        Computes the Bemis-Murcko scaffold for a molecule.
        Returns empty string representation or '_acyclic_' if no ring system exists.
        """
        try:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                return "_unknown_"
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            scaff_smiles = Chem.MolToSmiles(scaff, canonical=True)
            return scaff_smiles if scaff_smiles else "_acyclic_"
        except Exception:
            return "_unknown_"

    def parse_reaction_string(self, rxn_str: str) -> Tuple[List[str], List[str], List[str], Optional[str]]:
        """
        Splits reaction SMILES into (reactants, reagents, products, error).
        Handles reactants>reagents>products and reactants>>products, and Jin USPTO bond edit lines.
        """
        if not rxn_str or not isinstance(rxn_str, str):
            return [], [], [], "empty_input"

        s = rxn_str.strip()
        # Remove trailing bond edits e.g. ' 15-19;6-15;6-8'
        if " " in s:
            s = s.split(" ")[0].strip()

        # Remove CXSMILES annotations e.g. |f:2.3|
        if "|" in s:
            s = s.split("|")[0].strip()

        parts = s.split(">")
        if len(parts) == 3:
            r_raw, rg_raw, p_raw = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            r_raw, rg_raw, p_raw = parts[0], "", parts[1]
        elif len(parts) == 1:
            # Fallback if no delimiter
            return [], [], [], "malformed_syntax: missing '>' delimiter"
        else:
            return [], [], [], f"malformed_syntax: unexpected '>' count ({len(parts)-1})"

        reactants = [x.strip() for x in r_raw.split(".") if x.strip()]
        reagents = [x.strip() for x in rg_raw.split(".") if x.strip()]
        products = [x.strip() for x in p_raw.split(".") if x.strip()]

        return reactants, reagents, products, None

    def clean_reaction(
        self,
        rxn_str: Optional[str] = None,
        reactants_list: Optional[List[str]] = None,
        reagents_list: Optional[List[str]] = None,
        products_list: Optional[List[str]] = None,
        reaction_id: str = "",
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ProcessedReaction], Optional[str]]:
        """
        Cleans, canonicalizes, and verifies a reaction.
        Returns:
            (ProcessedReaction, None) on success
            (None, error_code) on failure
        """
        cond = conditions or {}

        # If full reaction string provided, parse components
        if rxn_str is not None:
            r_raw, rg_raw, p_raw, parse_err = self.parse_reaction_string(rxn_str)
            if parse_err:
                return None, parse_err
        else:
            r_raw = reactants_list or []
            rg_raw = reagents_list or []
            p_raw = products_list or []

        if not r_raw:
            return None, "empty_reactants"
        if not p_raw:
            return None, "empty_products"

        # Check single product requirement
        if self.require_single_product and len(p_raw) > 1:
            return None, f"multi_product: found {len(p_raw)} product molecules"

        # Clean products
        canon_products = []
        for p in p_raw:
            c, err = self.clean_molecule(p)
            if err or not c:
                return None, f"invalid_product: {err}"
            canon_products.append(c)

        # Clean reactants
        canon_reactants = []
        for r in r_raw:
            c, err = self.clean_molecule(r)
            if err or not c:
                return None, f"invalid_reactant: {err}"
            canon_reactants.append(c)

        # Sort reactants to ensure permutation invariance (A.B == B.A)
        canon_reactants.sort()

        # Clean reagents (preserve separately when available)
        canon_reagents = []
        for rg in rg_raw:
            c, err = self.clean_molecule(rg)
            if c:
                canon_reagents.append(c)
        canon_reagents.sort()

        # Extract Murcko scaffold on main product
        primary_product = canon_products[0]
        scaffold = self.extract_scaffold(primary_product)

        reactants_str = ".".join(canon_reactants)
        products_str = ".".join(canon_products)
        reagents_str = ".".join(canon_reagents)
        reaction_str = f"{reactants_str}>{reagents_str}>{products_str}"

        processed = ProcessedReaction(
            reaction_id=reaction_id or f"rxn_{abs(hash(reaction_str))}",
            reactants_smiles=reactants_str,
            products_smiles=products_str,
            reagents_smiles=reagents_str,
            reaction_smiles=reaction_str,
            product_scaffold=scaffold,
            is_valid=True,
            error_reason=None,
            conditions=cond,
        )

        return processed, None
