import React from 'react';
import { 
  Terminal, 
  AlertTriangle 
} from 'lucide-react';

export const DocumentationView: React.FC = () => {
  return (
    <div className="space-y-8 pb-16 max-w-6xl w-full mx-auto">
      
      {/* Header */}
      <div className="border-b border-[#E5E6DE] pb-5">
        <span className="text-xs font-mono uppercase tracking-wider text-[#87966A] font-semibold">
          Scientific Whitepaper & User Guide
        </span>
        <h1 className="text-3xl font-extrabold text-[#252820] tracking-tight mt-1">
          Documentation & Architectural Boundaries
        </h1>
        <p className="text-[#5E6356] text-sm mt-1 max-w-3xl">
          Methodological foundations, chemical assumptions, failure taxonomy, and exact CLI commands to reproduce all findings deterministically.
        </p>
      </div>

      {/* Critical Scientific Disclaimer - 3 Column Layout */}
      <section className="bg-white rounded-lg border border-[#E5E6DE] p-6 space-y-4 shadow-xs">
        <div className="flex items-center gap-2.5 text-[#B45309] font-mono font-bold text-xs uppercase tracking-wider">
          <AlertTriangle size={16} />
          <span>The Three Tiers of Chemical Verification</span>
        </div>
        <p className="text-xs text-[#5E6356] leading-relaxed max-w-3xl">
          ChemRxn-Intelligence enforces a strict epistemological boundary between computational graph processing and wet-lab physical reality:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <div className="p-4 rounded-md bg-[#FAF9F5] border border-[#F0F0EA] space-y-1.5 flex flex-col justify-between">
            <div>
              <div className="text-xs font-bold font-mono text-[#4D7C0F] uppercase tracking-wide">
                1. Chemical Syntax Validity
              </div>
              <p className="text-xs text-[#5E6356] leading-relaxed mt-2">
                Verifies that generated SMILES adheres to graph valence rules, ring closures, and aromaticity sanitization inside RDKit.
              </p>
            </div>
            <div className="text-[11px] font-mono font-semibold text-[#4D7C0F] pt-2 border-t border-[#E5E6DE]/60">
              Syntax ≠ Wet-lab Feasibility
            </div>
          </div>

          <div className="p-4 rounded-md bg-[#FAF9F5] border border-[#F0F0EA] space-y-1.5 flex flex-col justify-between">
            <div>
              <div className="text-xs font-bold font-mono text-[#252820] uppercase tracking-wide">
                2. Reference Agreement
              </div>
              <p className="text-xs text-[#5E6356] leading-relaxed mt-2">
                Calculates canonicalized string equality against historical patent claims extracted by Daniel Lowe from USPTO reaction literature.
              </p>
            </div>
            <div className="text-[11px] font-mono font-semibold text-[#5E6356] pt-2 border-t border-[#E5E6DE]/60">
              Concordance with Literature
            </div>
          </div>

          <div className="p-4 rounded-md bg-[#FAF9F5] border border-[#F0F0EA] space-y-1.5 flex flex-col justify-between">
            <div>
              <div className="text-xs font-bold font-mono text-[#B45309] uppercase tracking-wide">
                3. Wet-Lab Feasibility
              </div>
              <p className="text-xs text-[#5E6356] leading-relaxed mt-2">
                Never claimed without laboratory physical evidence. Reaction yields, kinetic activation barriers, and stereoisomeric purities require physical wet-lab confirmation.
              </p>
            </div>
            <div className="text-[11px] font-mono font-semibold text-[#B45309] pt-2 border-t border-[#E5E6DE]/60">
              Requires Laboratory Evidence
            </div>
          </div>
        </div>
      </section>

      {/* Architectural Boundaries & Limitations */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold text-[#252820]">
          Known Architectural Boundaries
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-white p-5 rounded-lg border border-[#E5E6DE] space-y-2">
            <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
              Incomplete Reaction Conditions
            </h3>
            <p className="text-xs text-[#5E6356] leading-relaxed">
              USPTO reaction SMILES routinely omit temperature, solvent ratios, catalysts, and reaction duration. The model predicts the constitutionally favored organic core product rather than dynamic kinetic distributions.
            </p>
          </div>

          <div className="bg-white p-5 rounded-lg border border-[#E5E6DE] space-y-2">
            <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
              Stereochemical Indeterminacy
            </h3>
            <p className="text-xs text-[#5E6356] leading-relaxed">
              Historical patent filings frequently report racemic mixtures without specifying wedge/dash chiral flags. Our evaluation reports constitutional exact-match in tandem with isomeric exact-match to separate connectivity from stereocenter ambiguity.
            </p>
          </div>

          <div className="bg-white p-5 rounded-lg border border-[#E5E6DE] space-y-2">
            <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
              Disconnected Molecule Graphs
            </h3>
            <p className="text-xs text-[#5E6356] leading-relaxed">
              Multi-reactant reactions (e.g., $A + B \rightarrow C$) are represented as disconnected molecular graph components. Message passing operates intramolecularly while graph readout aggregation compiles the mixture embedding.
            </p>
          </div>

          <div className="bg-white p-5 rounded-lg border border-[#E5E6DE] space-y-2">
            <h3 className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
              Abstention vs. Coverage Tradeoff
            </h3>
            <p className="text-xs text-[#5E6356] leading-relaxed">
              Setting the abstention threshold $\tau = 0.35$ eliminates wild hallucinations on out-of-domain inputs. On unfamiliar reagent pairs, the engine opts to abstain rather than emitting unreliable structures.
            </p>
          </div>
        </div>
      </section>

      {/* Deterministic Reproduction Commands */}
      <section className="bg-white rounded-lg border border-[#E5E6DE] p-6 space-y-4">
        <div className="flex items-center gap-2 text-xs font-mono font-bold uppercase tracking-wider text-[#252820]">
          <Terminal size={16} className="text-[#87966A]" />
          <span>Deterministic Reproduction Commands</span>
        </div>
        <p className="text-xs text-[#5E6356]">
          All benchmark tables, bootstrap error bars, and model checkpoints can be replicated using the provided scripts:
        </p>

        <div className="space-y-3 font-mono text-xs">
          <div className="space-y-1">
            <span className="text-[#898E81] text-[11px]"># 1. Run rigorous 5-model evaluation suite with bootstrap CIs</span>
            <pre className="p-3 bg-[#FAF9F5] border border-[#E5E6DE] rounded text-[#252820] overflow-x-auto">
python scripts/run_evaluation_framework.py --num-samples 20 --beam-size 4 --bootstrap-samples 1000
            </pre>
          </div>

          <div className="space-y-1">
            <span className="text-[#898E81] text-[11px]"># 2. Run automated test suite (77 unit & integration tests)</span>
            <pre className="p-3 bg-[#FAF9F5] border border-[#E5E6DE] rounded text-[#252820] overflow-x-auto">
python -m pytest tests/test_evaluation_framework.py -v
            </pre>
          </div>

          <div className="space-y-1">
            <span className="text-[#898E81] text-[11px]"># 3. Start the FastAPI backend server on port 8001</span>
            <pre className="p-3 bg-[#FAF9F5] border border-[#E5E6DE] rounded text-[#252820] overflow-x-auto">
uvicorn chemrxn.api.main:app --host 127.0.0.1 --port 8001 --reload
            </pre>
          </div>
        </div>
      </section>

    </div>
  );
};
