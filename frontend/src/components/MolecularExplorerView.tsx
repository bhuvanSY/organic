import React, { useState, useEffect } from 'react';
import type { PredictionResult, ReactionCompareResult } from '../types';
import { compareReactions } from '../api';
import { Molecule3DViewer } from './Molecule3DViewer';
import { 
  RotateCw,
  GitCompare,
  Info
} from 'lucide-react';

interface MolecularExplorerViewProps {
  currentPrediction: PredictionResult | null;
}

export const MolecularExplorerView: React.FC<MolecularExplorerViewProps> = ({
  currentPrediction,
}) => {
  const defaultReactant = currentPrediction?.reactants_canonical || currentPrediction?.reactants_input || 'CC(=O)Cl.c1ccccc1';
  const defaultProduct = currentPrediction?.candidates?.[0]?.canonical_smiles || currentPrediction?.candidates?.[0]?.product_smiles || 'CC(=O)c1ccccc1';

  const [reactantSmiles, setReactantSmiles] = useState<string>(defaultReactant);
  const [productSmiles, setProductSmiles] = useState<string>(defaultProduct);
  const [comparison, setComparison] = useState<ReactionCompareResult | null>(null);
  const [loadingCompare, setLoadingCompare] = useState<boolean>(false);

  // Compare reaction structures on mount or when SMILES changes
  useEffect(() => {
    let isMounted = true;
    if (!reactantSmiles.trim() || !productSmiles.trim()) {
      setComparison(null);
      return;
    }

    setLoadingCompare(true);
    compareReactions(reactantSmiles, productSmiles)
      .then((res) => {
        if (!isMounted) return;
        setComparison(res);
      })
      .catch((err) => {
        console.error('Reaction comparison error', err);
      })
      .finally(() => {
        if (isMounted) setLoadingCompare(false);
      });

    return () => {
      isMounted = false;
    };
  }, [reactantSmiles, productSmiles]);

  const handleSelectCandidate = (candSmi: string) => {
    setProductSmiles(candSmi);
  };

  return (
    <div className="space-y-8 pb-16">
      
      {/* Explorer Header */}
      <div className="border-b border-[#E5E6DE] pb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#87966A] font-semibold">
            Interactive Structural Reasoning
          </span>
          <h1 className="text-3xl font-extrabold text-[#252820] tracking-tight mt-1">
            Molecular Explorer & Reaction Centers
          </h1>
          <p className="text-[#5E6356] text-sm mt-1 max-w-2xl">
            Examine reactants and predicted product conformers in real interactive 3D WebGL via 3Dmol.js. Highlight reacting bond changes, conserved substructures (MCS), and mass balance deltas.
          </p>
        </div>

        {/* Candidate Switcher if Prediction is Active */}
        {currentPrediction?.candidates && currentPrediction.candidates.length > 0 && (
          <div className="flex items-center gap-1.5 bg-[#FAF9F5] p-1.5 rounded-lg border border-[#E5E6DE]">
            <span className="text-[11px] font-mono text-[#898E81] px-2">Candidate:</span>
            {currentPrediction.candidates.map((c, i) => {
              const smi = c.canonical_smiles || c.product_smiles;
              const isSelected = productSmiles === smi;
              return (
                <button
                  key={c.rank || i}
                  type="button"
                  onClick={() => handleSelectCandidate(smi)}
                  className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-all ${
                    isSelected ? 'bg-[#87966A] text-white shadow-sm' : 'bg-white text-[#5E6356] hover:text-[#252820]'
                  }`}
                >
                  #{c.rank || i + 1}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Input Inputs Bar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-white p-4 rounded-lg border border-[#E5E6DE]">
        <div>
          <label className="block text-xs font-bold font-mono uppercase tracking-wider text-[#5E6356] mb-1">
            Reactant Structure (SMILES)
          </label>
          <input
            type="text"
            value={reactantSmiles}
            onChange={(e) => setReactantSmiles(e.target.value)}
            className="w-full px-3 py-2 text-xs font-mono bg-[#FAF9F5] border border-[#E5E6DE] rounded focus:bg-white focus:border-[#87966A] focus:outline-none"
          />
        </div>

        <div>
          <label className="block text-xs font-bold font-mono uppercase tracking-wider text-[#5E6356] mb-1">
            Product Target Structure (SMILES)
          </label>
          <input
            type="text"
            value={productSmiles}
            onChange={(e) => setProductSmiles(e.target.value)}
            className="w-full px-3 py-2 text-xs font-mono bg-[#FAF9F5] border border-[#E5E6DE] rounded focus:bg-white focus:border-[#87966A] focus:outline-none"
          />
        </div>
      </div>

      {/* Side-by-Side Dual 3D Viewers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Left: Reactant 3D Viewer */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-[#252820] uppercase">Input Reactant Ensemble</span>
            <span className="text-[#898E81]">{comparison?.reactant_heavy_atoms || '—'} Heavy Atoms</span>
          </div>
          <Molecule3DViewer
            smiles={reactantSmiles}
            name="Reactant Ensemble (3D)"
            height={380}
            showControls={true}
          />
        </div>

        {/* Right: Predicted Product 3D Viewer */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-bold text-[#252820] uppercase">Predicted Product Molecule</span>
            <span className="text-[#898E81]">{comparison?.product_heavy_atoms || '—'} Heavy Atoms</span>
          </div>
          <Molecule3DViewer
            smiles={productSmiles}
            name="Predicted Product (3D)"
            height={380}
            highlightAtomIndices={comparison?.product_reaction_center_atom_indices || []}
            showControls={true}
          />
        </div>

      </div>

      {/* Reaction-Center Comparison Telemetry */}
      <div className="bg-white rounded-lg border border-[#E5E6DE] p-6 space-y-4">
        <div className="flex items-center justify-between border-b border-[#F0F0EA] pb-3">
          <div className="flex items-center gap-2">
            <GitCompare size={16} className="text-[#87966A]" />
            <h3 className="text-sm font-bold font-mono uppercase tracking-wider text-[#252820]">
              Reaction Center & Conserved Core Analysis (RDKit MCS)
            </h3>
          </div>
          {loadingCompare && (
            <span className="text-xs font-mono text-[#87966A] flex items-center gap-1">
              <RotateCw size={12} className="animate-spin" />
              <span>Analyzing substructures...</span>
            </span>
          )}
        </div>

        {comparison?.success ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E6DE]">
              <span className="text-[10px] font-mono uppercase text-[#898E81] block">CONSERVED ATOMS</span>
              <div className="text-xl font-bold font-['Manrope'] text-[#252820] mt-0.5">
                {comparison.num_conserved_atoms}
              </div>
              <span className="text-[11px] text-[#5E6356]">Core topological match</span>
            </div>

            <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E6DE]">
              <span className="text-[10px] font-mono uppercase text-[#898E81] block">CONSERVED BONDS</span>
              <div className="text-xl font-bold font-['Manrope'] text-[#252820] mt-0.5">
                {comparison.num_conserved_bonds}
              </div>
              <span className="text-[11px] text-[#5E6356]">Unchanged bond orders</span>
            </div>

            <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E6DE]">
              <span className="text-[10px] font-mono uppercase text-[#898E81] block">REACTION CENTERS</span>
              <div className="text-xl font-bold font-['Manrope'] text-[#4D7C0F] mt-0.5">
                {comparison.product_reaction_center_atom_indices.length} Atoms
              </div>
              <span className="text-[11px] text-[#5E6356]">Highlighted in product viewer</span>
            </div>

            <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E6DE]">
              <span className="text-[10px] font-mono uppercase text-[#898E81] block">MASS BALANCE DELTA</span>
              <div className="text-xl font-bold font-['Manrope'] text-[#252820] mt-0.5">
                {comparison.molecular_weight_delta > 0 ? `+${comparison.molecular_weight_delta}` : comparison.molecular_weight_delta} g/mol
              </div>
              <span className="text-[11px] text-[#5E6356]">Leaving group divergence</span>
            </div>
          </div>
        ) : (
          <div className="p-4 rounded bg-[#FAF9F5] text-xs text-[#5E6356]">
            Enter valid reactant and product SMILES strings to calculate Maximum Common Substructure (MCS) reaction differences.
          </div>
        )}

        {comparison?.mcs_smarts && (
          <div className="pt-2">
            <span className="text-[11px] font-mono uppercase text-[#898E81] block mb-1">
              Maximum Common Substructure SMARTS Core:
            </span>
            <code className="text-xs font-mono text-[#252820] bg-[#FAF9F5] p-2 rounded border border-[#E5E6DE] block overflow-x-auto">
              {comparison.mcs_smarts}
            </code>
          </div>
        )}

        <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E6DE] text-[11px] text-[#5E6356] flex items-start gap-2">
          <Info size={14} className="text-[#87966A] flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            {comparison?.disclaimer || 'Topological comparison highlights structural changes. Physical reaction yield and activation barriers require wet-lab experimental confirmation.'}
          </p>
        </div>
      </div>

    </div>
  );
};
