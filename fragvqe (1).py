#!/usr/bin/env python3
"""
FragVQE v1.1 - VQE Pipeline for Molecular Fragment Analysis
Author: Kantipudi Charan Sai Sree

Usage:
    python fragvqe.py fragment.xyz
    python fragvqe.py fragment.xyz --charge 0 --spin 1 --basis sto-3g
    python fragvqe.py fragment.xyz --depth 3 --restarts 5 --output results.txt
    python fragvqe.py fragment.xyz --hf-only

Install dependencies:
    pip install pyscf openfermion openfermionpyscf scipy numpy
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# ─────────────────────────────────────────────
# DEPENDENCY CHECKS
# ─────────────────────────────────────────────
try:
    from pyscf import gto, scf
except ImportError:
    print("[ERROR] PySCF not found. Install with: pip install pyscf")
    sys.exit(1)

try:
    from openfermion import MolecularData
    from openfermion.transforms import jordan_wigner, get_fermion_operator
    from openfermion.utils import count_qubits
    from openfermion.linalg import get_sparse_operator
except ImportError:
    print("[ERROR] OpenFermion not found. Install with: pip install openfermion openfermionpyscf")
    sys.exit(1)

try:
    from openfermionpyscf import run_pyscf
except ImportError:
    print("[ERROR] openfermionpyscf not found. Install with: pip install openfermionpyscf")
    sys.exit(1)


# ─────────────────────────────────────────────
# 1. XYZ PARSER
# ─────────────────────────────────────────────
def parse_xyz(filepath: str):
    """
    Parse a standard XYZ file.
    Format:
        Line 1 : number of atoms
        Line 2 : comment (ignored)
        Line 3+: symbol x y z
    Returns list of (symbol, x, y, z) tuples.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    atoms = []
    with open(path, "r") as f:
        lines = f.readlines()

    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        print("[ERROR] First line of XYZ file must be the atom count.")
        sys.exit(1)

    for line in lines[2: 2 + n_atoms]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append((symbol, x, y, z))

    if len(atoms) == 0:
        print("[ERROR] No atoms parsed from XYZ file.")
        sys.exit(1)

    return atoms


# ─────────────────────────────────────────────
# 2. BUILD HAMILTONIAN
# ─────────────────────────────────────────────
def build_hamiltonian(atoms, charge: int, spin: int, basis: str):
    """
    Build second-quantized Hamiltonian via PySCF + OpenFermion.
    Applies Jordan-Wigner transform to get qubit Hamiltonian.
    Returns qubit_hamiltonian, molecule object, n_qubits.
    """
    print(f"\n[INFO] Building Hamiltonian...")
    print(f"       Atoms     : {len(atoms)}")
    print(f"       Charge    : {charge}")
    print(f"       Spin mult : {spin}  (2S+1={spin})")
    print(f"       Basis     : {basis}")

    geometry = [(sym, (x, y, z)) for sym, x, y, z in atoms]

    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=spin,
        charge=charge,
        description="fragment"
    )

    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_fci=False,
        verbose=False
    )

    print(f"[INFO] HF Energy  : {molecule.hf_energy:.8f} Ha")
    print(f"[INFO] n_electrons: {molecule.n_electrons}")
    print(f"[INFO] n_orbitals : {molecule.n_orbitals}")

    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_op)
    n_qubits = count_qubits(qubit_hamiltonian)

    print(f"[INFO] Qubits (JW): {n_qubits}")

    return qubit_hamiltonian, molecule, n_qubits


# ─────────────────────────────────────────────
# 3. SINGLE QUBIT GATE (CORRECTED)
# Each basis state index is updated exactly once.
# gate[row, col] maps input col-bit to output row-bit.
# ─────────────────────────────────────────────
def apply_single_qubit_gate(state, qubit, n_qubits, gate):
    """
    Apply a 2x2 single-qubit gate to statevector.
    Each amplitude is updated exactly once — no double counting.
    """
    new_state = np.zeros_like(state)
    for i in range(len(state)):
        bit = (i >> qubit) & 1
        j = i ^ (1 << qubit)  # index with qubit bit flipped

        if bit == 0:
            # state[i] has qubit=0, state[j] has qubit=1
            new_state[i] += gate[0, 0] * state[i] + gate[0, 1] * state[j]
        else:
            # state[i] has qubit=1, state[j] has qubit=0
            new_state[i] += gate[1, 0] * state[j] + gate[1, 1] * state[i]

    return new_state


# ─────────────────────────────────────────────
# 4. CNOT GATE
# ─────────────────────────────────────────────
def apply_cnot(state, control, target, n_qubits):
    """
    Apply CNOT gate: flips target qubit when control qubit is |1>.
    """
    new_state = state.copy()
    for i in range(len(state)):
        if (i >> control) & 1:
            j = i ^ (1 << target)
            new_state[j] = state[i]
    return new_state


# ─────────────────────────────────────────────
# 5. RY ANSATZ (FIXED ENTANGLEMENT)
# Entanglement runs every layer so depth=1 is still expressive.
# ─────────────────────────────────────────────
def ry_ansatz(params, n_qubits, depth):
    """
    Hardware-efficient RY ansatz with CNOT entanglement.

    Structure per layer:
        - RY rotation on every qubit
        - CNOT chain (q0->q1, q1->q2, ...) runs EVERY layer

    FIX: entanglement now runs at every depth layer unconditionally
    (when n_qubits > 1), ensuring expressibility at all depths
    including depth=1. Previously it was skipped at the last layer.

    n_params = n_qubits * depth
    """
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0  # initialize to |00...0>

    idx = 0
    for d in range(depth):
        # RY rotation layer
        for q in range(n_qubits):
            theta = params[idx]
            idx += 1
            cos_t = np.cos(theta / 2)
            sin_t = np.sin(theta / 2)
            ry = np.array([[cos_t, -sin_t],
                           [sin_t,  cos_t]])
            state = apply_single_qubit_gate(state, q, n_qubits, ry)

        # Entanglement layer — runs every depth layer (fixed)
        # Condition: n_qubits > 1 only (not d < depth-1)
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                state = apply_cnot(state, q, q + 1, n_qubits)

    return state


# ─────────────────────────────────────────────
# 6. EXPECTATION VALUE (SPARSE)
# np.vdot handles complex conjugation correctly.
# Sparse H avoids building full dense matrix in memory.
# ─────────────────────────────────────────────
def expectation_sparse(state, sparse_H):
    """
    Compute <psi|H|psi> using sparse H.
    Efficient for large qubit counts.
    """
    return np.real(np.vdot(state, sparse_H @ state))


# ─────────────────────────────────────────────
# 7. FCI EXACT ENERGY (FULL DIAGONALIZATION)
# ─────────────────────────────────────────────
def fci_exact_energy(qubit_hamiltonian, n_qubits):
    """
    Compute exact ground state energy via full diagonalization.
    Only feasible for n_qubits <= ~20.
    """
    sparse_H = get_sparse_operator(qubit_hamiltonian, n_qubits)
    dense_H = sparse_H.toarray()
    eigenvalues = np.linalg.eigvalsh(dense_H)
    return float(np.min(eigenvalues))


# ─────────────────────────────────────────────
# 8. VQE OPTIMIZER
# ─────────────────────────────────────────────
def run_vqe(qubit_hamiltonian, n_qubits, depth=2, restarts=3):
    """
    Run VQE with random restarts to avoid local minima.
    Uses COBYLA optimizer — gradient-free, robust for noisy landscapes.
    maxiter=5000 handles larger fragments reliably.
    Returns best energy and optimized parameters.
    """
    sparse_H = get_sparse_operator(qubit_hamiltonian, n_qubits)
    n_params = n_qubits * depth
    best_energy = float('inf')
    best_params = None

    print(f"\n[VQE] Starting optimization...")
    print(f"      Qubits    : {n_qubits}")
    print(f"      Depth     : {depth}")
    print(f"      Params    : {n_params}")
    print(f"      Restarts  : {restarts}")
    print(f"      Optimizer : COBYLA (maxiter=5000)")

    for r in range(restarts):
        init_params = np.random.uniform(-np.pi, np.pi, n_params)

        def cost(params):
            state = ry_ansatz(params, n_qubits, depth)
            return expectation_sparse(state, sparse_H)

        result = minimize(
            cost,
            init_params,
            method='COBYLA',
            options={
                'maxiter': 5000,
                'rhobeg': 0.5
            }
        )

        status = "converged" if result.success else "not converged"
        print(f"      Restart {r+1}/{restarts}: Energy = {result.fun:.8f} Ha  ({status})")

        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x.copy()

    print(f"\n[VQE] Best energy across all restarts: {best_energy:.8f} Ha")
    return best_energy, best_params


# ─────────────────────────────────────────────
# 9. OUTPUT WRITER
# ─────────────────────────────────────────────
def write_output(results: dict, output_file: str = None):
    """
    Print results summary to console.
    Optionally save to a text file.
    """
    fci = results.get('fci_energy')
    vqe = results['vqe_energy']

    if fci is not None and not np.isnan(fci):
        err_ha   = abs(vqe - fci)
        err_kcal = err_ha * 627.509
        fci_line   = f"  FCI Energy      : {fci:.8f} Ha"
        error_line = f"  VQE - FCI error : {err_ha:.2e} Ha  ({err_kcal:.4f} kcal/mol)"
        corr_line  = f"  Correlation E   : {fci - results['hf_energy']:.8f} Ha"
    else:
        fci_line   = "  FCI Energy      : skipped (--no-fci)"
        error_line = "  VQE - FCI error : N/A"
        corr_line  = "  Correlation E   : N/A"

    lines = [
        "=" * 60,
        "              FragVQE OUTPUT SUMMARY",
        "=" * 60,
        f"  Input file      : {results['input_file']}",
        f"  Basis set       : {results['basis']}",
        f"  Charge          : {results['charge']}",
        f"  Spin (2S+1)     : {results['spin']}",
        f"  n_atoms         : {results['n_atoms']}",
        f"  n_electrons     : {results['n_electrons']}",
        f"  n_orbitals      : {results['n_orbitals']}",
        f"  Qubits (JW)     : {results['n_qubits']}",
        "-" * 60,
        f"  HF Energy       : {results['hf_energy']:.8f} Ha",
        fci_line,
        f"  VQE Energy      : {vqe:.8f} Ha",
        "-" * 60,
        error_line,
        corr_line,
        f"  Total wall time : {results['wall_time']:.2f} seconds",
        "=" * 60,
    ]

    output_str = "\n".join(lines)
    print("\n" + output_str)

    if output_file:
        with open(output_file, "w") as f:
            f.write(output_str + "\n")
        print(f"\n[INFO] Results saved to: {output_file}")


# ─────────────────────────────────────────────
# 10. MAIN CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        prog="fragvqe",
        description="VQE energy calculation for molecular fragments from XYZ files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python fragvqe.py fragment1.xyz
  python fragvqe.py fragment1.xyz --charge -1 --spin 1 --basis 6-31g
  python fragvqe.py fragment1.xyz --depth 3 --restarts 5 --output results.txt
  python fragvqe.py fragment1.xyz --hf-only
  python fragvqe.py fragment1.xyz --no-fci
        """
    )

    parser.add_argument(
        "xyz_file",
        type=str,
        help="Path to input XYZ geometry file"
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Molecular charge (default: 0)"
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=1,
        help="Spin multiplicity 2S+1 (default: 1 = singlet)"
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set (default: sto-3g). Use sto-3g for speed, 6-31g for accuracy."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Ansatz circuit depth (default: 2)"
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=3,
        help="Number of VQE random restarts (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to this file (e.g. results.txt)"
    )
    parser.add_argument(
        "--hf-only",
        action="store_true",
        help="Run Hartree-Fock only, skip VQE (fast sanity check)"
    )
    parser.add_argument(
        "--no-fci",
        action="store_true",
        help="Skip FCI exact diagonalization (saves memory for large fragments)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("                    FragVQE v1.1")
    print("     VQE Pipeline for Molecular Fragment Analysis")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Parse XYZ
    print(f"\n[INFO] Reading: {args.xyz_file}")
    atoms = parse_xyz(args.xyz_file)
    print(f"[INFO] Parsed {len(atoms)} atoms successfully")

    # Step 2: Build Hamiltonian
    qubit_hamiltonian, molecule, n_qubits = build_hamiltonian(
        atoms, args.charge, args.spin, args.basis
    )

    # HF only mode
    if args.hf_only:
        print(f"\n[INFO] --hf-only flag set. Skipping VQE and FCI.")
        print(f"[RESULT] HF Energy : {molecule.hf_energy:.8f} Ha")
        print(f"[RESULT] Qubits    : {n_qubits}")
        return

    # Memory warning for large systems
    if n_qubits > 20:
        print(f"\n[WARNING] {n_qubits} qubits detected.")
        print(f"[WARNING] Statevector size = {2**n_qubits} amplitudes.")
        print(f"[WARNING] This may be very slow or exceed available memory.")
        print(f"[WARNING] Consider --hf-only or a smaller basis set (sto-3g).")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("[INFO] Exiting.")
            sys.exit(0)

    # Step 3: FCI exact energy
    fci_energy = None
    if not args.no_fci:
        print(f"\n[INFO] Computing FCI exact energy (full diagonalization)...")
        fci_energy = fci_exact_energy(qubit_hamiltonian, n_qubits)
        print(f"[INFO] FCI Energy : {fci_energy:.8f} Ha")

    # Step 4: VQE
    vqe_energy, best_params = run_vqe(
        qubit_hamiltonian,
        n_qubits,
        depth=args.depth,
        restarts=args.restarts
    )

    wall_time = time.time() - start_time

    # Step 5: Output
    results = {
        "input_file":  args.xyz_file,
        "basis":       args.basis,
        "charge":      args.charge,
        "spin":        args.spin,
        "n_atoms":     len(atoms),
        "n_electrons": molecule.n_electrons,
        "n_orbitals":  molecule.n_orbitals,
        "n_qubits":    n_qubits,
        "hf_energy":   molecule.hf_energy,
        "fci_energy":  fci_energy if fci_energy is not None else float('nan'),
        "vqe_energy":  vqe_energy,
        "wall_time":   wall_time
    }

    write_output(results, args.output)


if __name__ == "__main__":
    main()
