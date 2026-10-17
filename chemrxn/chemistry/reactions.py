"""
Reaction SMILES parsing and standardization.
Supports both standard reaction SMILES (reactants>reagents>products) and Jin USPTO format.
"""

from typing import Dict, List, Optional, Tuple
from chemrxn.chemistry.smiles import canonicalize_smiles, is_valid_smiles, strip_atom_mapping


def parse_reaction_smiles(reaction_smiles: str) -> Dict[str, List[str]]:
    """
    Parses a reaction SMILES into components: reactants, reagents, and products.
    Handles 'reactants>reagents>products' or 'reactants>>products'.
    Also strips Jin-style trailing metadata (e.g. ' 15-19;6-15;6-8').
    """
    cleaned = reaction_smiles.strip()
    # Strip any trailing bond change annotations (space separated)
    if " " in cleaned:
        cleaned = cleaned.split(" ")[0].strip()

    parts = cleaned.split(">")
    if len(parts) == 3:
        reactants_str, reagents_str, products_str = parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        reactants_str, reagents_str, products_str = parts[0], "", parts[1]
    elif len(parts) == 1:
        # If no '>' separator, assume reactants only
        reactants_str, reagents_str, products_str = parts[0], "", ""
    else:
        raise ValueError(f"Malformed reaction SMILES string: {reaction_smiles}")

    reactants = [r.strip() for r in reactants_str.split(".") if r.strip()]
    reagents = [rg.strip() for rg in reagents_str.split(".") if rg.strip()]
    products = [p.strip() for p in products_str.split(".") if p.strip()]

    return {
        "reactants": reactants,
        "reagents": reagents,
        "products": products,
    }


def clean_reaction_smiles(reaction_smiles: str, remove_mapping: bool = True) -> Optional[str]:
    """
    Canonicalizes all components of a reaction SMILES and reassembles them.
    Returns None if any required reactant or product is chemically invalid.
    """
    try:
        parsed = parse_reaction_smiles(reaction_smiles)
        
        canon_reactants = []
        for r in parsed["reactants"]:
            c = canonicalize_smiles(r, remove_mapping=remove_mapping)
            if c:
                canon_reactants.append(c)
        
        canon_reagents = []
        for rg in parsed["reagents"]:
            c = canonicalize_smiles(rg, remove_mapping=remove_mapping)
            if c:
                canon_reagents.append(c)
                
        canon_products = []
        for p in parsed["products"]:
            c = canonicalize_smiles(p, remove_mapping=remove_mapping)
            if c:
                canon_products.append(c)

        if not canon_reactants or not canon_products:
            return None

        # Sort reactants to ensure permutation invariance
        canon_reactants.sort()
        canon_reagents.sort()
        canon_products.sort()

        r_str = ".".join(canon_reactants)
        rg_str = ".".join(canon_reagents)
        p_str = ".".join(canon_products)

        return f"{r_str}>{rg_str}>{p_str}"
    except Exception:
        return None
