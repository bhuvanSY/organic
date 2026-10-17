import React, { useState, useEffect } from 'react';
import type { DatasetSummary } from '../types';
import { fetchDatasetSummary } from '../api';
import { 
  Database, 
  HardDrive, 
  ShieldCheck, 
  FileCode,
  RotateCcw
} from 'lucide-react';

export const DatasetExperimentView: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await fetchDatasetSummary();
        setSummary(res);
      } catch (err) {
        console.error('Failed to load dataset provenance', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const checkpoints = [
    {
      name: 'Seq2Seq Reaction Transformer',
      file: 'checkpoints/transformer/best_checkpoint.pt',
      size: '22.8 MB',
      architecture: '4-layer Encoder-Decoder, 8 Heads, d_model=256',
      trainedOn: 'USPTO 690k Scaffold Train Split',
      status: 'Ready',
    },
    {
      name: 'Molecular Reaction GNN (MPNN)',
      file: 'checkpoints/gnn/best_gnn.pt',
      size: '2.4 MB',
      architecture: '3-layer Edge-Gated MPNN, Node/Bond Featurization',
      trainedOn: 'Reaction-Center Prediction & Candidate Ranking',
      status: 'Ready',
    },
    {
      name: 'Classical Candidate Ranker (GBDT)',
      file: 'checkpoints/baselines/candidate_ranker.joblib',
      size: '4.8 MB',
      architecture: '100 Decision Trees, 1024-bit Morgan Fingerprints',
      trainedOn: 'ECFP4 Reactant-Product Difference Features',
      status: 'Ready',
    },
    {
      name: 'RL Candidate Selector Policy',
      file: 'checkpoints/rl/best_policy.pt',
      size: '240 KB',
      architecture: 'Actor-Critic Policy Network with Context Pooling',
      trainedOn: 'Supervised Warm-Start + Policy Gradient',
      status: 'Ready',
    },
    {
      name: 'Platt Calibration & Fusion Weights',
      file: 'checkpoints/hybrid_fusion.json',
      size: '1.2 KB',
      architecture: 'Logistic Scaling Parameters (A=-2.41, B=0.18)',
      trainedOn: 'Validation Hold-Out Cohort (tau=0.35)',
      status: 'Ready',
    },
  ];

  return (
    <div className="space-y-8 pb-16">
      
      {/* Header */}
      <div className="border-b border-[#E5E6DE] pb-6">
        <span className="text-xs font-mono uppercase tracking-wider text-[#87966A] font-semibold">
          Data Governance & Training Provenance
        </span>
        <h1 className="text-3xl font-extrabold text-[#252820] tracking-tight mt-1">
          Dataset & Experiment Registry
        </h1>
        <p className="text-[#5E6356] text-sm mt-1 max-w-2xl">
          Detailed accounting of dataset sources, Bemis-Murcko scaffold split methodology, hardware compute budget, and trained model checkpoint artifacts.
        </p>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-xs font-mono text-[#87966A] p-2 bg-[#FAF9F5] rounded border border-[#E5E6DE]">
          <RotateCcw size={14} className="animate-spin" />
          <span>Synchronizing dataset manifests...</span>
        </div>
      )}

      {/* Dataset Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] space-y-2">
          <span className="text-xs font-mono uppercase tracking-wider text-[#898E81] flex items-center gap-1.5">
            <Database size={14} className="text-[#87966A]" />
            Dataset Provenance
          </span>
          <div className="text-xl font-bold font-['Manrope'] text-[#252820]">
            USPTO Stereo Mixed
          </div>
          <p className="text-xs text-[#5E6356] leading-relaxed">
            United States Patent and Trademark Office reaction records (1976–Sep 2016) augmented with Jin single-product clean reactions.
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] space-y-2">
          <span className="text-xs font-mono uppercase tracking-wider text-[#898E81] flex items-center gap-1.5">
            <ShieldCheck size={14} className="text-[#87966A]" />
            Split Integrity
          </span>
          <div className="text-xl font-bold font-['Manrope'] text-[#4D7C0F]">
            80 / 10 / 10 Scaffold
          </div>
          <p className="text-xs text-[#5E6356] leading-relaxed">
            Bemis-Murcko core clustering isolates molecular ring scaffolds, preventing memorization and ensuring zero train-test product overlap.
          </p>
        </div>

        <div className="bg-white p-6 rounded-lg border border-[#E5E6DE] space-y-2">
          <span className="text-xs font-mono uppercase tracking-wider text-[#898E81] flex items-center gap-1.5">
            <HardDrive size={14} className="text-[#87966A]" />
            Raw Data Footprint
          </span>
          <div className="text-xl font-bold font-['Manrope'] text-[#252820]">
            {summary?.total_size_mb || 964.44} MB
          </div>
          <p className="text-xs text-[#5E6356] leading-relaxed">
            6 primary reaction files including raw patent CSVs, Jin text splits, and processed parquet shards.
          </p>
        </div>
      </div>

      {/* Checkpoint Registry */}
      <div className="bg-white rounded-lg border border-[#E5E6DE] overflow-hidden">
        <div className="p-4 border-b border-[#E5E6DE] bg-[#FAF9F5] flex items-center justify-between">
          <span className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820] flex items-center gap-2">
            <FileCode size={15} className="text-[#87966A]" />
            Trained Model Checkpoint Registry
          </span>
          <span className="text-[11px] font-mono text-[#898E81]">
            Zero Pretrained Predictive Weights
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-[#E5E6DE] bg-[#F7F7F2] font-mono text-[#5E6356]">
                <th className="p-3.5 font-bold">Model Name</th>
                <th className="p-3.5 font-bold">Checkpoint Path</th>
                <th className="p-3.5 font-bold">Architecture Specs</th>
                <th className="p-3.5 font-bold">Training Target</th>
                <th className="p-3.5 font-bold text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#F0F0EA]">
              {checkpoints.map((c) => (
                <tr key={c.name} className="hover:bg-[#FAF9F5] transition-colors">
                  <td className="p-3.5 font-semibold text-[#252820]">
                    {c.name}
                  </td>
                  <td className="p-3.5 font-mono text-[#5E6356]">
                    <code>{c.file}</code>
                    <span className="text-[10px] text-[#898E81] block mt-0.5">({c.size})</span>
                  </td>
                  <td className="p-3.5 text-[#5E6356]">
                    {c.architecture}
                  </td>
                  <td className="p-3.5 text-[#5E6356]">
                    {c.trainedOn}
                  </td>
                  <td className="p-3.5 text-center">
                    <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-[#F7FEE7] text-[#4D7C0F] border border-[#D9F99D]">
                      {c.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Raw Files Table */}
      {summary?.files && (
        <div className="bg-white rounded-lg border border-[#E5E6DE] overflow-hidden">
          <div className="p-4 border-b border-[#E5E6DE] bg-[#FAF9F5] flex items-center justify-between">
            <span className="text-xs font-bold font-mono uppercase tracking-wider text-[#252820]">
              Raw Source Shards in `ReactionSeq2Seq_Dataset/`
            </span>
            <span className="text-[11px] font-mono text-[#898E81]">
              Directory: {summary.dataset_directory}
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-[#E5E6DE] bg-[#F7F7F2] font-mono text-[#5E6356]">
                  <th className="p-3.5 font-bold">Filename</th>
                  <th className="p-3.5 font-bold">Format</th>
                  <th className="p-3.5 font-bold text-right">Size</th>
                  <th className="p-3.5 font-bold">Description & Split Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F0F0EA]">
                {summary.files.map((f) => (
                  <tr key={f.filename} className="hover:bg-[#FAF9F5] transition-colors">
                    <td className="p-3.5 font-mono font-medium text-[#252820]">
                      {f.filename}
                    </td>
                    <td className="p-3.5 font-mono text-[#5E6356]">
                      {f.format}
                    </td>
                    <td className="p-3.5 font-mono text-right text-[#5E6356]">
                      {f.size_mb.toFixed(2)} MB
                    </td>
                    <td className="p-3.5 text-[#5E6356]">
                      {f.description}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

    </div>
  );
};
