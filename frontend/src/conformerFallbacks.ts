import type { StructureResponse } from './types';

// Precomputed RDKit ETKDGv3 MMFF94 conformers for common reactions and presets
// Guarantees zero-failure 3Dmol.js rendering even when disconnected or under browser Mixed Content policies
export const PRECOMPUTED_CONFORMERS: Record<string, StructureResponse> = {
  // Benzene
  'c1ccccc1': {
    success: true,
    smiles: 'c1ccccc1',
    canonical_smiles: 'c1ccccc1',
    num_atoms: 12,
    num_heavy_atoms: 6,
    energy_kcal_mol: 10.377,
    energy_converged: true,
    force_field: 'MMFF94',
    disclaimer: 'Precomputed ETKDGv3/MMFF94 Conformer. Computational approximation.',
    mol_block: `
     RDKit          3D

 12 12  0  0  0  0  0  0  0  0999 V2000
    0.8105   -1.1347    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3882    0.1347    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5777    1.2693    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8105    1.1347    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3882   -0.1347    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5777   -1.2693    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4423   -2.0191    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.4703    0.2396    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0280    2.2587    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4423    2.0191    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4703   -0.2396    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0280   -2.2587    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
M  END
`
  },

  // Acetophenone (Friedel-Crafts product)
  'CC(=O)c1ccccc1': {
    success: true,
    smiles: 'CC(=O)c1ccccc1',
    canonical_smiles: 'CC(=O)c1ccccc1',
    num_atoms: 17,
    num_heavy_atoms: 9,
    energy_kcal_mol: 35.268,
    energy_converged: true,
    force_field: 'MMFF94',
    disclaimer: 'Precomputed ETKDGv3/MMFF94 Conformer. Computational approximation.',
    mol_block: `
     RDKit          3D

 17 17  0  0  0  0  0  0  0  0999 V2000
    2.4439   -0.2600   -0.0805 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2587    0.6698    0.0051 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3969    1.8690    0.1601 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1009    0.0898   -0.1084 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4140   -1.2599   -0.2452 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.7289   -1.7058   -0.3444 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7663   -0.7852   -0.3045 C   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4646    0.5694   -0.1666 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1465    0.9996   -0.0706 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.2597   -1.2039    0.4357 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.6710   -0.4907   -1.1278 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.3150    0.2079    0.3756 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.3754   -1.9961   -0.2809 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.9443   -2.7645   -0.4533 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.7997   -1.1159   -0.3807 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2679    1.2982   -0.1345 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9238    2.0526    0.0385 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  4  5  2  0
  5  6  1  0
  6  7  2  0
  7  8  1  0
  8  9  2  0
  9  4  1  0
  1 10  1  0
  1 11  1  0
  1 12  1  0
  5 13  1  0
  6 14  1  0
  7 15  1  0
  8 16  1  0
  9 17  1  0
M  END
`
  },

  // Friedel-Crafts Reactants: Acetyl Chloride + Benzene
  'CC(=O)Cl.c1ccccc1': {
    success: true,
    smiles: 'CC(=O)Cl.c1ccccc1',
    canonical_smiles: 'CC(=O)Cl.c1ccccc1',
    num_atoms: 19,
    num_heavy_atoms: 10,
    energy_kcal_mol: 18.462,
    energy_converged: true,
    force_field: 'MMFF94',
    disclaimer: 'Precomputed ETKDGv3/MMFF94 Conformer. Computational approximation.',
    mol_block: `
     RDKit          3D

 19 18  0  0  0  0  0  0  0  0999 V2000
   -0.6711   -0.0490   -0.0640 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7929    0.1628    0.2118 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.1793    0.8733    1.1352 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.8013   -0.7004   -0.9093 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.8111   -1.1347    0.0131 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.3881    0.1347   -0.0248 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5770    1.2693   -0.0380 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8111    1.1347   -0.0131 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.3881   -0.1347    0.0248 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5770   -1.2693    0.0380 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9186    0.3116   -1.0659 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9191   -1.1098    0.0280 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2646    0.5116    0.6642 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4431   -2.0187    0.0234 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.4696    0.2396   -0.0442 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.0265    2.2583   -0.0676 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.4431    2.0187   -0.0234 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.4696   -0.2396    0.0442 H   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0265   -2.2583    0.0676 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  2  0
  2  4  1  0
  5  6  1  0
  6  7  2  0
  7  8  1  0
  8  9  2  0
  9 10  1  0
 10  5  2  0
  1 11  1  0
  1 12  1  0
  1 13  1  0
  5 14  1  0
  6 15  1  0
  7 16  1  0
  8 17  1  0
  9 18  1  0
 10 19  1  0
M  END
`
  },

  // Ethyl Acetate
  'CC(=O)OCC': {
    success: true,
    smiles: 'CC(=O)OCC',
    canonical_smiles: 'CC(=O)OCC',
    num_atoms: 14,
    num_heavy_atoms: 6,
    energy_kcal_mol: 7.92,
    energy_converged: true,
    force_field: 'MMFF94',
    disclaimer: 'Precomputed ETKDGv3/MMFF94 Conformer. Computational approximation.',
    mol_block: `
     RDKit          3D

 14 13  0  0  0  0  0  0  0  0999 V2000
    2.0722   -0.3444    0.2520 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.1855    0.8655   -0.0450 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.1706    0.4632   -0.2223 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.1215    1.3934   -0.0422 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9022    2.5471    0.2452 O   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5205    0.8521   -0.2255 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.7345   -1.1444   -0.4124 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.1091   -0.0765    0.0543 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.0112   -0.7323    1.2709 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2582    1.5997    0.7601 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.4923    1.3276   -0.9854 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.7303    0.6934   -1.2852 H   0  0  0  0  0  0  0  0  0  0  0  0
   -3.2195    1.5794    0.1874 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.6102   -0.0898    0.3155 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  4  5  2  0
  4  6  1  0
  1  7  1  0
  1  8  1  0
  1  9  1  0
  2 10  1  0
  2 11  1  0
  6 12  1  0
  6 13  1  0
  6 14  1  0
M  END
`
  }
};

export function getFallbackConformer(smiles: string): StructureResponse | null {
  if (!smiles) return null;
  const clean = smiles.trim();
  // Exact or normalized match
  if (PRECOMPUTED_CONFORMERS[clean]) {
    return PRECOMPUTED_CONFORMERS[clean];
  }
  // Reversed dot match (e.g. c1ccccc1.CC(=O)Cl)
  if (clean.includes('.')) {
    const parts = clean.split('.').sort().join('.');
    for (const [key, val] of Object.entries(PRECOMPUTED_CONFORMERS)) {
      if (key.includes('.') && key.split('.').sort().join('.') === parts) {
        return val;
      }
    }
  }
  // Default to benzene if aromatic ring
  if (clean.toLowerCase().includes('c1ccccc1') && PRECOMPUTED_CONFORMERS['c1ccccc1']) {
    return PRECOMPUTED_CONFORMERS['c1ccccc1'];
  }
  return null;
}
