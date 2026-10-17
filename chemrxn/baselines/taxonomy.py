"""
Organic Reaction Taxonomy and Transformation Classification.
Implements the 10 standard organic reaction superclasses (Schneider et al., USPTO).
Maps reactions to chemical classes based on molecular structure and transformation analysis.
"""

from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

REACTION_CLASSES = [
    "Heteroatom_Alkylation_Arylation",  # 0: Ether, ester, amine alkylation
    "Acylation_and_Related",            # 1: Amide coupling, esterification
    "C_C_Bond_Formation",               # 2: Cross-coupling, Grignard, aldol
    "Heterocycle_Formation",            # 3: Ring closure forming heteroaromatics
    "Protections",                      # 4: Boc, Cbz, silyl, benzyl protection
    "Deprotections",                    # 5: Deprotection, ester hydrolysis
    "Reductions",                       # 6: Nitro->amine, carbonyl->alcohol, hydrogenation
    "Oxidations",                       # 7: Alcohol->carbonyl, sulfoxidation
    "Functional_Group_Interconversion", # 8: Halogenation, azidation, FGI
    "Functional_Group_Addition",        # 9: Addition across multiple bonds
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(REACTION_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(REACTION_CLASSES)}


class ReactionClassifierTagger:
    """
    Infers reaction transformation class using chemical SMARTS and atom count changes.
    Used to assign empirical ground truth reaction classes for classical ML training.
    """

    def __init__(self):
        # Key transformation patterns
        self.amide_bond = Chem.MolFromSmarts("[C;!R](=O)[N;!R]")
        self.nitro_group = Chem.MolFromSmarts("[$([N+](=O)[O-])]")
        self.primary_amine = Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")
        self.carbonyl = Chem.MolFromSmarts("[CX3]=[OX1]")
        self.alcohol = Chem.MolFromSmarts("[OX2H]")
        self.halide = Chem.MolFromSmarts("[F,Cl,Br,I]")

    def classify_reaction(
        self,
        reactants_smiles: str,
        products_smiles: str,
        reagents_smiles: str = "",
    ) -> str:
        """
        Infers reaction class from reactants and product structures.
        """
        try:
            r_mol = Chem.MolFromSmiles(reactants_smiles)
            p_mol = Chem.MolFromSmiles(products_smiles)
            if r_mol is None or p_mol is None:
                return "Functional_Group_Interconversion"

            # 1. Ring analysis for heterocycles
            r_rings = rdMolDescriptors.CalcNumHeterocycles(r_mol)
            p_rings = rdMolDescriptors.CalcNumHeterocycles(p_mol)
            if p_rings > r_rings:
                return "Heterocycle_Formation"

            # 2. Reduction: nitro to amine or carbonyl reduction
            if r_mol.HasSubstructMatch(self.nitro_group) and not p_mol.HasSubstructMatch(self.nitro_group):
                if p_mol.HasSubstructMatch(self.primary_amine):
                    return "Reductions"

            r_h = Descriptors.ExactMolWt(r_mol)
            p_h = Descriptors.ExactMolWt(p_mol)

            # 3. Acylation / Amide coupling
            r_amides = len(r_mol.GetSubstructMatches(self.amide_bond))
            p_amides = len(p_mol.GetSubstructMatches(self.amide_bond))
            if p_amides > r_amides:
                return "Acylation_and_Related"

            # 4. Oxidation: alcohol to carbonyl
            r_alcohols = len(r_mol.GetSubstructMatches(self.alcohol))
            p_alcohols = len(p_mol.GetSubstructMatches(self.alcohol))
            r_carbonyls = len(r_mol.GetSubstructMatches(self.carbonyl))
            p_carbonyls = len(p_mol.GetSubstructMatches(self.carbonyl))
            if r_alcohols > p_alcohols and p_carbonyls > r_carbonyls:
                return "Oxidations"
            elif r_carbonyls > p_carbonyls and p_alcohols > r_alcohols:
                return "Reductions"

            # 5. Carbon-Carbon bond formation
            r_c = sum(1 for a in r_mol.GetAtoms() if a.GetSymbol() == "C")
            p_c = sum(1 for a in p_mol.GetAtoms() if a.GetSymbol() == "C")
            if p_c > r_c:
                return "C_C_Bond_Formation"

            # 6. Deprotection / Protections
            if "boc" in reagents_smiles.lower() or "CC(C)(C)OC(=O)" in reactants_smiles:
                if "CC(C)(C)OC(=O)" not in products_smiles:
                    return "Deprotections"
                return "Protections"

            # 7. Heteroatom Alkylation / Arylation
            r_halides = len(r_mol.GetSubstructMatches(self.halide))
            p_halides = len(p_mol.GetSubstructMatches(self.halide))
            if r_halides > p_halides and p_mol.GetNumHeavyAtoms() >= r_mol.GetNumHeavyAtoms():
                return "Heteroatom_Alkylation_Arylation"

            # 8. Halogenation / FGI
            if p_halides > r_halides:
                return "Functional_Group_Interconversion"

            # Default
            return "Heteroatom_Alkylation_Arylation"
        except Exception:
            return "Functional_Group_Interconversion"
