"""
Environment and Foundation Verification Script for ChemRxn-Intelligence.
Checks Python runtime, PyTorch CUDA/CPU fallback, RDKit chemistry engine,
seed determinism, scratch model initialization, and dataset files.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.abspath("."))


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def verify_python():
    print_header("1. Python Environment Check")
    version = sys.version.split()[0]
    executable = sys.executable
    print(f"Python Executable: {executable}")
    print(f"Python Version   : {version}")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"FAIL: Python version {version} is below 3.10 requirement.")
        return False
    print("PASS: Python version requirement satisfied.")
    return True


def verify_pytorch_and_hardware():
    print_header("2. PyTorch & Hardware Acceleration Check")
    print(f"PyTorch Version   : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available    : {cuda_avail}")

    if cuda_avail:
        device_count = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        print(f"NVIDIA GPU Count  : {device_count}")
        print(f"Primary Device    : {dev_name}")
        print(f"Total VRAM        : {vram_gb} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"SM MultiProcessors: {props.multi_processor_count}")

        # CUDA tensor computation test
        x = torch.randn(256, 256, device="cuda:0")
        y = torch.matmul(x, x)
        print(f"CUDA Tensor Matmul: Successful (shape: {list(y.shape)})")
    else:
        print("CUDA is NOT available. Falling back to CPU.")
        cpu_t = torch.randn(128, 128)
        y = torch.matmul(cpu_t, cpu_t)
        print(f"CPU Tensor Matmul : Successful (shape: {list(y.shape)})")

    # CPU Fallback verification
    cpu_device = torch.device("cpu")
    cpu_x = torch.ones(4, 4, device=cpu_device)
    print(f"CPU Fallback Test : Successful (device: {cpu_x.device})")
    print("PASS: PyTorch and compute device verification passed.")
    return True


def verify_rdkit_chemistry():
    print_header("3. RDKit Chemistry Engine Check")
    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from chemrxn.chemistry.fingerprints import compute_morgan_fingerprint, compute_tanimoto_similarity
        from chemrxn.chemistry.smiles import canonicalize_smiles, compute_molecular_descriptors, strip_atom_mapping

        print(f"RDKit Version     : {rdkit.__version__}")

        # Test canonicalization & atom-map removal
        mapped_smiles = "[CH3:1][C:2](=[O:3])[OH:4]"
        stripped = strip_atom_mapping(mapped_smiles)
        canonical = canonicalize_smiles(stripped)
        print(f"Input SMILES      : {mapped_smiles}")
        print(f"Stripped Atom-Map : {stripped}")
        print(f"Canonical SMILES  : {canonical}")
        assert canonical == "CC(=O)O", f"Expected 'CC(=O)O', got '{canonical}'"

        # Test molecular descriptors
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"
        desc = compute_molecular_descriptors(aspirin)
        assert desc is not None
        print(f"Aspirin Descriptors: Formula={desc['molecular_formula']}, MW={desc['molecular_weight']}, LogP={desc['logp']}")

        # Test Morgan fingerprints and Tanimoto similarity
        fp1 = compute_morgan_fingerprint("CC(=O)O")
        fp2 = compute_morgan_fingerprint("CCC(=O)O")
        sim = compute_tanimoto_similarity("CC(=O)O", "CCC(=O)O")
        print(f"Morgan FP Shape   : {fp1.shape} bits")
        print(f"Tanimoto Similarity (Acetic vs Propionic Acid): {sim:.4f}")
        assert 0.0 < sim < 1.0

        print("PASS: RDKit chemistry operations verified.")
        return True
    except Exception as e:
        print(f"FAIL: Chemistry verification failed: {e}")
        return False


def verify_seed_determinism():
    print_header("4. Deterministic Seed Handling Check")
    from chemrxn.utils.seed import seed_everything

    seed_everything(42)
    t1 = torch.randn(5, 5)
    np1 = np.random.rand(5)

    seed_everything(42)
    t2 = torch.randn(5, 5)
    np2 = np.random.rand(5)

    torch_match = torch.equal(t1, t2)
    numpy_match = np.array_equal(np1, np2)
    print(f"PyTorch Seed Match: {torch_match}")
    print(f"NumPy Seed Match  : {numpy_match}")

    if torch_match and numpy_match:
        print("PASS: Deterministic seeding operates reproducibly.")
        return True
    else:
        print("FAIL: Deterministic seed mismatch.")
        return False


def verify_scratch_models():
    print_header("5. Scratch Neural Network Verification")
    from chemrxn.models.fingerprint_mlp import FingerprintReactionMLP
    from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer

    # 1. Fingerprint MLP check
    mlp = FingerprintReactionMLP(input_dim=2048, hidden_dim=512, output_dim=2048, num_layers=2)
    params_mlp = mlp.count_parameters()
    print(f"Fingerprint MLP Params: Total={params_mlp['total_parameters']:,}, Trainable={params_mlp['trainable_parameters']:,}")
    dummy_fp = torch.randn(4, 2048)
    out_mlp = mlp(dummy_fp)
    print(f"Fingerprint Forward Pass: Shape={list(out_mlp.shape)}, Value Range=[{out_mlp.min().item():.3f}, {out_mlp.max().item():.3f}]")

    # 2. Seq2Seq Transformer check
    transformer = Seq2SeqReactionTransformer(vocab_size=64, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2)
    params_tf = transformer.count_parameters()
    print(f"Transformer Params    : Total={params_tf['total_parameters']:,}, Trainable={params_tf['trainable_parameters']:,}")
    dummy_src = torch.randint(0, 64, (2, 20))
    dummy_tgt = torch.randint(0, 64, (2, 20))
    out_tf = transformer(dummy_src, dummy_tgt)
    print(f"Transformer Forward   : Shape={list(out_tf.shape)}")

    print("PASS: Neural architectures instantiated from scratch with random weights.")
    return True


def verify_dataset_integrity():
    print_header("6. Dataset Files Integrity Check")
    from chemrxn.data.dataset import load_reaction_samples

    dataset_dir = Path("ReactionSeq2Seq_Dataset")
    if not dataset_dir.exists():
        print(f"FAIL: Dataset directory {dataset_dir} does not exist.")
        return False

    files = list(dataset_dir.iterdir())
    print(f"Found {len(files)} dataset files:")
    for f in sorted(files, key=lambda x: x.name):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name:52s} : {size_mb:7.2f} MB")

    # Test loading 10 records from valid dataset
    valid_file = dataset_dir / "US_patents_1976-Sep2016_1product_reactions_valid.csv"
    if valid_file.exists():
        records = load_reaction_samples(valid_file, max_samples=10)
        print(f"\nSample load test (US Patents valid): Successfully parsed {len(records)} reaction records.")
        if records:
            r0 = records[0]
            print(f"  Reactants: {r0.reactants[:60]}...")
            print(f"  Products : {r0.products[:60]}...")

    jin_valid = dataset_dir / "Jin_USPTO_1product_valid.txt"
    if jin_valid.exists():
        jin_records = load_reaction_samples(jin_valid, max_samples=10)
        print(f"Sample load test (Jin USPTO valid): Successfully parsed {len(jin_records)} reaction records.")

    print("PASS: Dataset directory and reaction parsers verified.")
    return True


def main():
    print("=" * 70)
    print("   CHEMRXN-INTELLIGENCE: SETUP & FOUNDATION VERIFICATION")
    print("=" * 70)

    checks = [
        ("Python Environment", verify_python),
        ("PyTorch & Hardware", verify_pytorch_and_hardware),
        ("RDKit Chemistry", verify_rdkit_chemistry),
        ("Seed Determinism", verify_seed_determinism),
        ("Scratch Neural Models", verify_scratch_models),
        ("Dataset Integrity", verify_dataset_integrity),
    ]

    all_passed = True
    results = {}
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results[name] = "PASSED" if passed else "FAILED"
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = f"ERROR: {e}"
            all_passed = False

    print_header("FINAL VERIFICATION SUMMARY")
    for name, status in results.items():
        status_symbol = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"  {status_symbol:7s} {name:25s} : {status}")

    print("=" * 70)
    if all_passed:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY.")
        print("Environment and foundational modules are ready for research.")
        return 0
    else:
        print("SOME VERIFICATION CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
