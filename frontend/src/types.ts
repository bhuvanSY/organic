export interface HealthInfo {
  status: string;
  version: string;
  project_name: string;
  torch_version: string;
  rdkit_version: string;
  cuda_available: boolean;
  device_name: string;
  device_type: string;
}

export interface HardwareInfo {
  cuda_available: boolean;
  torch_version: string;
  device_type: string;
  device_name: string;
  total_memory_gb?: number | string;
  allocated_memory_mb?: number;
  reserved_memory_mb?: number;
  multi_processor_count?: number;
  compute_capability?: string;
  supports_fp16?: boolean;
  supports_bf16?: boolean;
  cpu_count?: number;
}

export interface MolecularDescriptors {
  smiles: string;
  molecular_formula?: string;
  molecular_weight?: number;
  logp?: number;
  tpsa?: number;
  num_heavy_atoms?: number;
  num_rotatable_bonds?: number;
  num_h_donors?: number;
  num_h_acceptors?: number;
  num_aromatic_rings?: number;
  num_saturated_rings?: number;
}

export interface SMILESValidationResult {
  is_valid: boolean;
  input_smiles: string;
  canonical_smiles?: string;
  stripped_smiles?: string;
  descriptors?: MolecularDescriptors;
  error?: string;
}

export interface CandidateProduct {
  rank: number;
  product_smiles: string;
  canonical_smiles?: string;
  is_chemically_valid: boolean;
  is_syntactically_valid?: boolean;
  confidence_score: number;
  calibrated_confidence?: number;
  tanimoto_to_reactants?: number;
  validity_note?: string;
  model_provenance?: string;
  descriptors?: Record<string, unknown>;
}

export interface PredictionResult {
  success: boolean;
  reactants_input: string;
  reactants_canonical?: string;
  reactant_descriptors?: Record<string, unknown>;
  model_status: string;
  is_trained: boolean;
  candidates: CandidateProduct[];
  hardware_device: string;
  abstained?: boolean;
  abstention_reason?: string;
  top_calibrated_confidence?: number;
  epistemic_uncertainty?: number;
  latency_ms?: number;
  error?: string;
}

export interface DatasetFile {
  filename: string;
  size_mb: number;
  format: string;
  description: string;
}

export interface DatasetSummary {
  dataset_directory: string;
  total_files: number;
  total_size_mb: number;
  files: DatasetFile[];
  train_record_estimate: string;
}

export interface ModelBenchmarkSummary {
  model_name: string;
  top1_acc_pct: number;
  top1_ci_lower?: number;
  top1_ci_upper?: number;
  top3_acc_pct?: number;
  top5_acc_pct?: number;
  stereo_aware_acc_pct?: number;
  valid_smiles_rate_pct?: number;
  abstention_rate_pct?: number;
  precision_at_coverage_pct?: number;
  ece: number;
  median_latency_ms: number;
  throughput_rxn_sec?: number;
  parameter_count: string;
}

export interface EvaluationReportData {
  metadata: {
    evaluator_version: string;
    test_samples_evaluated: number;
    beam_size: number;
    n_bootstrap: number;
    hardware_device: string;
    peak_gpu_memory_mb: number;
    abstention_threshold: number;
    disclaimer: string;
  };
  model_summaries: ModelBenchmarkSummary[];
  validity_breakdown: {
    valid_rate: number;
    total_evaluated: number;
    categories: Record<string, number>;
    category_percentages: Record<string, number>;
  };
  scaffold_generalization: {
    train_scaffold_count: number;
    test_reactions_evaluated: number;
    exact_product_leakage_count: number;
    exact_product_leakage_rate: number;
    seen_scaffold_samples: number;
    seen_scaffold_accuracy: number;
    unseen_scaffold_samples: number;
    unseen_scaffold_accuracy: number;
  };
}

export interface Atom3DCoord {
  index: number;
  symbol: string;
  atomic_number: number;
  x: number;
  y: number;
  z: number;
  formal_charge: number;
  is_aromatic: boolean;
  hybridization: string;
}

export interface StructureResponse {
  success: boolean;
  smiles: string;
  canonical_smiles?: string;
  num_atoms?: number;
  num_heavy_atoms?: number;
  mol_block?: string;
  sdf?: string;
  energy_kcal_mol?: number | null;
  energy_converged?: boolean;
  force_field?: string;
  atoms?: Atom3DCoord[];
  error?: string;
  disclaimer: string;
}

export interface ReactionCompareResult {
  success: boolean;
  reactant_smiles: string;
  product_smiles: string;
  mcs_smarts: string;
  num_conserved_atoms: number;
  num_conserved_bonds: number;
  reactant_heavy_atoms: number;
  product_heavy_atoms: number;
  molecular_weight_delta: number;
  product_reaction_center_atom_indices: number[];
  mapping_confidence: string;
  disclaimer: string;
  error?: string;
}

export interface ModelCheckpointInfo {
  name: string;
  checkpoint_path: string;
  exists: boolean;
  size_mb?: number | null;
  architecture: string;
  trained_status: string;
}

export interface ModelStatusResponse {
  status: string;
  device: string;
  cuda_available: boolean;
  models: ModelCheckpointInfo[];
  ready_for_inference: boolean;
}

export type ActiveRoute = 'landing' | 'workbench' | 'results' | 'explorer' | 'benchmark' | 'dataset' | 'docs';
