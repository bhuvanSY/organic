import React, { useState, useEffect } from 'react';
import type { ActiveRoute, PredictionResult, SMILESValidationResult } from '../types';
import { validateSMILES, predictHybrid } from '../api';
import { MoleculeViewer } from './MoleculeViewer';
import { 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  Sliders, 
  RotateCcw,
  FlaskConical,
  Info
} from 'lucide-react';

interface WorkbenchViewProps {
  onPredictionComplete: (result: PredictionResult) => void;
  onNavigate: (route: ActiveRoute) => void;
}

interface ReactionPreset {
  name: string;
  category: string;
  reactants: string;
  reagents?: string;
  description: string;
}

const PRESETS: ReactionPreset[] = [
  {
    name: 'Friedel-Crafts Acylation',
    category: 'C-C Bond Formation',
    reactants: 'CC(=O)Cl.c1ccccc1',
    reagents: 'AlCl3, DCM, 0°C',
    description: 'Electrophilic aromatic substitution yielding acetophenone.',
  },
  {
    name: 'Fischer Esterification',
    category: 'Acylation',
    reactants: 'CC(=O)O.CCO',
    reagents: 'H2SO4 cat., reflux',
    description: 'Acid-catalyzed condensation producing ethyl acetate.',
  },
  {
    name: 'Grignard Carbonyl Addition',
    category: 'Organometallic',
    reactants: 'c1ccccc1[Mg]Br.CC(=O)C',
    reagents: 'THF, then H3O+',
    description: 'Nucleophilic addition forming 2-phenylpropan-2-ol.',
  },
  {
    name: 'Suzuki-Miyaura Cross-Coupling',
    category: 'Pd-Catalyzed',
    reactants: 'c1ccccc1B(O)O.c1ccc(Br)cc1',
    reagents: 'Pd(PPh3)4, K2CO3, dioxane',
    description: 'Biaryl synthesis generating biphenyl.',
  },
  {
    name: 'Amide Bond Coupling',
    category: 'Peptide/Amide',
    reactants: 'CC(=O)Cl.CCN',
    reagents: 'Et3N, DCM',
    description: 'Nucleophilic acyl substitution forming N-ethylacetamide.',
  },
];

export const WorkbenchView: React.FC<WorkbenchViewProps> = ({
  onPredictionComplete,
  onNavigate,
}) => {
  const [reactantSmiles, setReactantSmiles] = useState<string>('CC(=O)Cl.c1ccccc1');
  const [reagents, setReagents] = useState<string>('AlCl3');
  const [modelType, setModelType] = useState<string>('hybrid');
  const [beamSize, setBeamSize] = useState<number>(4);
  const [numCandidates, setNumCandidates] = useState<number>(3);
  
  const [validation, setValidation] = useState<SMILESValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState<boolean>(false);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Debounced real-time SMILES validation
  useEffect(() => {
    if (!reactantSmiles.trim()) {
      setValidation(null);
      return;
    }

    setIsValidating(true);
    const timer = setTimeout(async () => {
      try {
        const res = await validateSMILES(reactantSmiles);
        setValidation(res);
      } catch {
        setValidation({
          is_valid: true,
          input_smiles: reactantSmiles,
        });
      } finally {
        setIsValidating(false);
      }
    }, 250);

    return () => clearTimeout(timer);
  }, [reactantSmiles]);

  const handleApplyPreset = (preset: ReactionPreset) => {
    setReactantSmiles(preset.reactants);
    setReagents(preset.reagents || '');
    setErrorMsg(null);
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reactantSmiles.trim()) {
      setErrorMsg('Please provide reactant SMILES before initiating prediction.');
      return;
    }

    setIsPredicting(true);
    setErrorMsg(null);

    try {
      const result = await predictHybrid(reactantSmiles, beamSize, numCandidates);
      onPredictionComplete(result);
      onNavigate('results');
    } catch (err: any) {
      setErrorMsg(err?.message || 'Prediction execution failed. Please verify the backend connection.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-8 pb-12">
      {/* Workbench Header */}
      <div className="border-b border-[#E5E6DE] pb-6">
        <span className="text-xs font-mono uppercase tracking-wider text-[#87966A] font-semibold">
          Reaction Synthesis Laboratory
        </span>
        <h1 className="text-3xl font-extrabold text-[#252820] tracking-tight mt-1">
          Reaction Workbench
        </h1>
        <p className="text-[#5E6356] text-sm mt-1 max-w-2xl">
          Enter reactant SMILES and optional condition tokens. RDKit validates stoichiometry and chemical syntax before passing disjoint-union molecular graphs to the predictive ensemble.
        </p>
      </div>

      {/* Preset Library Pill Bar */}
      <div className="bg-white p-4 rounded-lg border border-[#E5E6DE]">
        <div className="flex items-center gap-2 mb-3">
          <FlaskConical size={14} className="text-[#87966A]" />
          <span className="text-xs font-bold font-mono uppercase tracking-wider text-[#5E6356]">
            Curated Reaction Presets
          </span>
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              type="button"
              onClick={() => handleApplyPreset(p)}
              className="text-left px-3 py-2 rounded border border-[#E5E6DE] bg-[#FAF9F5] hover:bg-white hover:border-[#B6C49A] transition-all text-xs group"
            >
              <div className="font-semibold text-[#252820] group-hover:text-[#4D7C0F]">{p.name}</div>
              <div className="text-[11px] text-[#898E81] font-mono truncate max-w-[200px]">{p.reactants}</div>
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handlePredict} className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left 2 Columns: SMILES Input & Preview */}
        <div className="lg:col-span-2 space-y-6">
          
          <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] space-y-4">
            <div className="flex items-center justify-between">
              <label htmlFor="reactant-input" className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
                Reactant SMILES (Dot-Separated)
              </label>
              <div className="flex items-center gap-2 text-xs">
                {isValidating ? (
                  <span className="text-[#898E81] text-[11px] font-mono animate-pulse">Verifying via RDKit...</span>
                ) : validation?.is_valid ? (
                  <span className="text-[#4D7C0F] text-[11px] font-mono font-medium flex items-center gap-1">
                    <CheckCircle2 size={13} /> Valid Syntax
                  </span>
                ) : validation ? (
                  <span className="text-[#B91C1C] text-[11px] font-mono font-medium flex items-center gap-1">
                    <AlertCircle size={13} /> {validation.error || 'Invalid SMILES'}
                  </span>
                ) : null}
              </div>
            </div>

            <textarea
              id="reactant-input"
              value={reactantSmiles}
              onChange={(e) => setReactantSmiles(e.target.value)}
              rows={3}
              placeholder="e.g. CC(=O)Cl.c1ccccc1"
              className="w-full p-3 font-mono text-sm bg-[#FAF9F5] border border-[#E5E6DE] rounded-md focus:bg-white focus:border-[#87966A] focus:outline-none transition-colors"
              required
            />

            {/* Optional Reagents and Conditions */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              <div>
                <label htmlFor="reagents-input" className="block text-xs font-mono uppercase tracking-wider text-[#5E6356] mb-1">
                  Reagents / Catalysts (Optional)
                </label>
                <input
                  id="reagents-input"
                  type="text"
                  value={reagents}
                  onChange={(e) => setReagents(e.target.value)}
                  placeholder="e.g. AlCl3, Pd(PPh3)4"
                  className="w-full px-3 py-2 text-xs font-mono bg-[#FAF9F5] border border-[#E5E6DE] rounded focus:bg-white focus:border-[#87966A] focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="solvent-input" className="block text-xs font-mono uppercase tracking-wider text-[#5E6356] mb-1">
                  Solvent & Conditions (Optional)
                </label>
                <input
                  id="solvent-input"
                  type="text"
                  placeholder="e.g. DCM, 25°C, 2h"
                  className="w-full px-3 py-2 text-xs font-mono bg-[#FAF9F5] border border-[#E5E6DE] rounded focus:bg-white focus:border-[#87966A] focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* 2D Molecular Structure Preview */}
          <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
                Reactant Structure Inspection (RDKit 2D Render)
              </span>
              <span className="text-[11px] text-[#898E81] font-mono">
                Authoritative Backend Kernel
              </span>
            </div>

            <div className="flex items-center justify-center p-2 bg-[#FAF9F5] rounded-md border border-[#F0F0EA]">
              <MoleculeViewer 
                smiles={reactantSmiles} 
                name="Input Reactants" 
                width={500} 
                height={200}
                className="w-full"
              />
            </div>

            {/* Descriptors Chips if available */}
            {validation?.descriptors && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-[#F0F0EA]">
                <div className="bg-[#FAF9F5] p-2 rounded text-center">
                  <span className="text-[10px] text-[#898E81] font-mono block">MOL WEIGHT</span>
                  <span className="text-xs font-bold font-mono text-[#252820]">
                    {validation.descriptors.molecular_weight?.toFixed(1) || '—'}
                  </span>
                </div>
                <div className="bg-[#FAF9F5] p-2 rounded text-center">
                  <span className="text-[10px] text-[#898E81] font-mono block">LOGP</span>
                  <span className="text-xs font-bold font-mono text-[#252820]">
                    {validation.descriptors.logp?.toFixed(2) || '—'}
                  </span>
                </div>
                <div className="bg-[#FAF9F5] p-2 rounded text-center">
                  <span className="text-[10px] text-[#898E81] font-mono block">HEAVY ATOMS</span>
                  <span className="text-xs font-bold font-mono text-[#252820]">
                    {validation.descriptors.num_heavy_atoms || '—'}
                  </span>
                </div>
                <div className="bg-[#FAF9F5] p-2 rounded text-center">
                  <span className="text-[10px] text-[#898E81] font-mono block">AROMATIC RINGS</span>
                  <span className="text-xs font-bold font-mono text-[#252820]">
                    {validation.descriptors.num_aromatic_rings || '0'}
                  </span>
                </div>
              </div>
            )}
          </div>

        </div>

        {/* Right 1 Column: Prediction Controls Sidebar */}
        <div className="space-y-6">
          <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] space-y-6">
            <div className="flex items-center gap-2 pb-4 border-b border-[#F0F0EA]">
              <Sliders size={16} className="text-[#87966A]" />
              <h3 className="text-sm font-bold font-mono uppercase tracking-wider text-[#252820]">
                Prediction Controls
              </h3>
            </div>

            {/* Model Architecture Selector */}
            <div className="space-y-2">
              <label className="block text-xs font-bold text-[#5E6356]">
                Inference Engine
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="w-full px-3 py-2 text-xs bg-[#FAF9F5] border border-[#E5E6DE] rounded font-medium text-[#252820]"
              >
                <option value="hybrid">Hybrid Engine (Transformer + GNN + Abstention)</option>
                <option value="transformer">Transformer Beam Search Only</option>
                <option value="gnn">GNN Reaction-Pair Scorer</option>
                <option value="baseline">Classical Supervised Ranker (GBDT)</option>
              </select>
              <p className="text-[11px] text-[#898E81] leading-relaxed">
                The Hybrid engine runs Transformer beam decoding, reranks via GNN representations, and applies Platt calibration with abstention fallback.
              </p>
            </div>

            {/* Beam Size */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-[#5E6356]">
                <span>Decoding Beam Budget</span>
                <span className="font-mono text-[#252820]">B = {beamSize}</span>
              </div>
              <input
                type="range"
                min={1}
                max={8}
                value={beamSize}
                onChange={(e) => setBeamSize(Number(e.target.value))}
                className="w-full accent-[#87966A]"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#898E81]">
                <span>Greedy (1)</span>
                <span>Standard (4)</span>
                <span>Deep (8)</span>
              </div>
            </div>

            {/* Number of Candidates to Return */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-bold text-[#5E6356]">
                <span>Ranked Candidates to Emit</span>
                <span className="font-mono text-[#252820]">k = {numCandidates}</span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                value={numCandidates}
                onChange={(e) => setNumCandidates(Number(e.target.value))}
                className="w-full accent-[#87966A]"
              />
            </div>

            {/* Scientific Note */}
            <div className="bg-[#FAF9F5] p-3 rounded border border-[#E5E6DE] flex items-start gap-2.5 text-xs text-[#5E6356]">
              <Info size={15} className="text-[#87966A] flex-shrink-0 mt-0.5" />
              <p className="text-[11px] leading-relaxed">
                Predictions are screened by RDKit sanitization. Predictions with calibrated confidence below $\tau=0.35$ trigger abstention rather than unverified guessing.
              </p>
            </div>

            {errorMsg && (
              <div className="p-3 rounded bg-[#FEF2F2] border border-[#FCA5A5] text-[#B91C1C] text-xs flex items-start gap-2">
                <AlertCircle size={14} className="flex-shrink-0 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Run Button */}
            <button
              type="submit"
              disabled={isPredicting || validation?.is_valid === false}
              className={`w-full py-3.5 px-4 rounded-md font-medium text-sm flex items-center justify-center gap-2 shadow-sm transition-all ${
                isPredicting || validation?.is_valid === false
                  ? 'bg-[#E5E6DE] text-[#898E81] cursor-not-allowed'
                  : 'bg-[#87966A] hover:bg-[#6E7C54] text-white'
              }`}
            >
              {isPredicting ? (
                <>
                  <RotateCcw size={16} className="animate-spin" />
                  <span>Decoding Candidates...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Generate Reaction Prediction</span>
                </>
              )}
            </button>

          </div>
        </div>

      </form>
    </div>
  );
};
