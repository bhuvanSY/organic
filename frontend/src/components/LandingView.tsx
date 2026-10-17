import React from 'react';
import type { ActiveRoute } from '../types';
import { 
  ArrowRight, 
  ShieldCheck, 
  Binary, 
  Network, 
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

interface LandingViewProps {
  onNavigate: (route: ActiveRoute) => void;
}

export const LandingView: React.FC<LandingViewProps> = ({ onNavigate }) => {
  return (
    <div className="space-y-16 pb-16">
      
      {/* Hero Section */}
      <section className="relative pt-8 pb-12 border-b border-[#E5E6DE]">
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#F2F5ED] border border-[#B6C49A]/50 text-xs font-mono font-medium text-[#4D7C0F] mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-[#87966A]" />
            Rigorous Scientific Reaction Modeling — USPTO Stereo Mixed Split
          </div>

          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-[#252820] leading-[1.12] mb-6">
            Predict forward organic reactions with calibrated structural confidence.
          </h1>

          <p className="text-lg sm:text-xl text-[#5E6356] leading-relaxed mb-8 max-w-3xl">
            ChemRxn-Intelligence pairs an independently trained scratch Seq2Seq Transformer with a Molecular Graph Neural Network and a Platt-calibrated candidate selector. Chemical validity is strictly enforced via RDKit rather than raw neural heuristics.
          </p>

          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={() => onNavigate('workbench')}
              className="inline-flex items-center gap-2.5 px-6 py-3.5 rounded-md bg-[#87966A] hover:bg-[#6E7C54] text-white font-medium text-sm shadow-sm transition-all"
            >
              <span>Open Reaction Workbench</span>
              <ArrowRight size={16} />
            </button>

            <button
              onClick={() => onNavigate('benchmark')}
              className="inline-flex items-center gap-2 px-5 py-3.5 rounded-md bg-white hover:bg-[#F0F0EA] text-[#252820] border border-[#E5E6DE] font-medium text-sm transition-all"
            >
              <span>Inspect Benchmark Studio</span>
            </button>

            <button
              onClick={() => onNavigate('docs')}
              className="inline-flex items-center gap-2 px-4 py-3.5 text-[#5E6356] hover:text-[#252820] text-sm font-medium transition-colors"
            >
              <span>Methodology & Disclaimers</span>
            </button>
          </div>
        </div>

        {/* Empirical Metrics Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-14 pt-8 border-t border-[#E5E6DE]">
          <div className="bg-white p-4 rounded-lg border border-[#E5E6DE]">
            <span className="text-xs font-mono uppercase tracking-wider text-[#898E81]">Locked Test Set</span>
            <div className="text-2xl font-bold font-['Manrope'] text-[#252820] mt-1">86,348</div>
            <p className="text-xs text-[#5E6356] mt-0.5">Scaffold-isolated reaction pairs</p>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#E5E6DE]">
            <span className="text-xs font-mono uppercase tracking-wider text-[#898E81]">Exact Leakage</span>
            <div className="text-2xl font-bold font-['Manrope'] text-[#4D7C0F] mt-1">0.00%</div>
            <p className="text-xs text-[#5E6356] mt-0.5">Zero train-test product overlap</p>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#E5E6DE]">
            <span className="text-xs font-mono uppercase tracking-wider text-[#898E81]">RL Selector ECE</span>
            <div className="text-2xl font-bold font-['Manrope'] text-[#252820] mt-1">0.0014</div>
            <p className="text-xs text-[#5E6356] mt-0.5">Expected calibration error</p>
          </div>

          <div className="bg-white p-4 rounded-lg border border-[#E5E6DE]">
            <span className="text-xs font-mono uppercase tracking-wider text-[#898E81]">Inference Latency</span>
            <div className="text-2xl font-bold font-['Manrope'] text-[#252820] mt-1">1.98 ms</div>
            <p className="text-xs text-[#5E6356] mt-0.5">RTX 4050 FP16 acceleration</p>
          </div>
        </div>
      </section>

      {/* Core Architectural Pillars */}
      <section className="space-y-8">
        <div>
          <span className="text-xs font-mono uppercase tracking-wider text-[#87966A] font-semibold">
            System Design
          </span>
          <h2 className="text-2xl sm:text-3xl font-bold text-[#252820] mt-1">
            Engineered for Chemical Integrity, Not Merely Pattern Matching
          </h2>
          <p className="text-[#5E6356] mt-2 max-w-2xl">
            Raw language models frequently hallucinate non-physical valences or misplace stereocenters. ChemRxn-Intelligence separates generative search from physical verification.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Pillar 1 */}
          <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-md bg-[#F2F5ED] text-[#87966A] flex items-center justify-center mb-4">
                <Binary size={20} />
              </div>
              <h3 className="text-lg font-bold text-[#252820] mb-2">Scratch Seq2Seq Transformer</h3>
              <p className="text-sm text-[#5E6356] leading-relaxed">
                4-layer encoder-decoder Transformer with 8 attention heads, custom SMILES sub-token vocabulary, sinusoidal positional encodings, and length-penalized beam search decoding ($B=4$).
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-[#F0F0EA] flex items-center justify-between text-xs font-mono text-[#898E81]">
              <span>5.57M Parameters</span>
              <span>PyTorch Native</span>
            </div>
          </div>

          {/* Pillar 2 */}
          <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-md bg-[#F2F5ED] text-[#87966A] flex items-center justify-center mb-4">
                <Network size={20} />
              </div>
              <h3 className="text-lg font-bold text-[#252820] mb-2">Molecular Graph Neural Network</h3>
              <p className="text-sm text-[#5E6356] leading-relaxed">
                Message-Passing Neural Network (MPNN) representing atoms as nodes and chemical bonds as edges. Employs gated graph convolutions to pinpoint reacting centers and score reaction pair plausibility.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-[#F0F0EA] flex items-center justify-between text-xs font-mono text-[#898E81]">
              <span>568K Parameters</span>
              <span>Disjoint-Union Batching</span>
            </div>
          </div>

          {/* Pillar 3 */}
          <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-md bg-[#F2F5ED] text-[#87966A] flex items-center justify-center mb-4">
                <ShieldCheck size={20} />
              </div>
              <h3 className="text-lg font-bold text-[#252820] mb-2">Calibrated Selection & Abstention</h3>
              <p className="text-sm text-[#5E6356] leading-relaxed">
                Ensemble fusion with Platt-scaled validation probabilities and an experimental Actor-Critic RL policy. Abstains when epistemic uncertainty exceeds threshold $\tau=0.35$ to avoid false positives.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-[#F0F0EA] flex items-center justify-between text-xs font-mono text-[#898E81]">
              <span>ECE = 0.0014</span>
              <span>Zero Invalid Selections</span>
            </div>
          </div>
        </div>
      </section>

      {/* Scientific Integrity Callout */}
      <section className="bg-white rounded-lg border border-[#E5E6DE] p-8">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded bg-[#FFFBEB] text-[#B45309] flex items-center justify-center flex-shrink-0 mt-0.5">
            <AlertTriangle size={20} />
          </div>
          <div className="space-y-3">
            <h3 className="text-lg font-bold text-[#252820]">
              Explicit Scientific Disclaimers & Rigor Guarantees
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
              <div className="space-y-1">
                <h4 className="text-xs font-bold font-mono uppercase tracking-wider text-[#4D7C0F] flex items-center gap-1.5">
                  <CheckCircle2 size={13} />
                  1. Syntactic Validity
                </h4>
                <p className="text-xs text-[#5E6356] leading-relaxed">
                  Confirms that a SMILES string satisfies graph valence rules and sanitization in RDKit. Does not imply physical product yield.
                </p>
              </div>

              <div className="space-y-1">
                <h4 className="text-xs font-bold font-mono uppercase tracking-wider text-[#4D7C0F] flex items-center gap-1.5">
                  <CheckCircle2 size={13} />
                  2. Reference Agreement
                </h4>
                <p className="text-xs text-[#5E6356] leading-relaxed">
                  Measures canonical exact-match against historical patent literature (USPTO). Reflects empirical ground truth records.
                </p>
              </div>

              <div className="space-y-1">
                <h4 className="text-xs font-bold font-mono uppercase tracking-wider text-[#B45309] flex items-center gap-1.5">
                  <AlertTriangle size={13} />
                  3. Wet-Lab Feasibility
                </h4>
                <p className="text-xs text-[#5E6356] leading-relaxed">
                  Never asserted without laboratory experimental evidence. Thermodynamic stability and activation barriers require physical validation.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

    </div>
  );
};
