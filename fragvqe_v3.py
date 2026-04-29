#!/usr/bin/env python3
"""
FragVQE v3.2
Correct Active Space + Stable VQE (No double counting)
"""

import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from pathlib import Path

from openfermion import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf


# ─────────────────────────────────────────────
# XYZ PARSER
# ─────────────────────────────────────────────
def parse_xyz(file):
    lines = open(file).readlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2:2+n]:
        s,x,y,z = line.split()
        atoms.append((s,float(x),float(y),float(z)))
    return atoms


# ─────────────────────────────────────────────
# ACTIVE SPACE (CORRECT)
# ─────────────────────────────────────────────
def build_active_hamiltonian(atoms, active_orbitals=6):

    mol = MolecularData(
        geometry=[(s,(x,y,z)) for s,x,y,z in atoms],
        basis="sto-3g",
        multiplicity=1,
        charge=0
    )

    mol = run_pyscf(mol, run_scf=True, run_fci=False)

    n_orb = mol.n_orbitals
    n_elec = mol.n_electrons

    # HOMO-LUMO centered selection
    n_occ = n_elec // 2
    start = max(0, n_occ - active_orbitals//2)
    stop  = min(n_orb, start + active_orbitals)

    active_idx = list(range(start, stop))
    core_idx   = list(range(start))

    print(f"[ACTIVE] Orbitals: {active_idx}")

    # IMPORTANT: this already includes core contributions
    hamiltonian = mol.get_molecular_hamiltonian(
        occupied_indices=core_idx,
        active_indices=active_idx
    )

    fermion = get_fermion_operator(hamiltonian)
    qubit = jordan_wigner(fermion)

    n_qubits = 2 * len(active_idx)
    print(f"[INFO] Active qubits: {n_qubits}")

    return qubit, mol, n_qubits


# ─────────────────────────────────────────────
# GATES
# ─────────────────────────────────────────────
def apply_gate(state, q, gate):
    state = state.reshape(-1,2,2**q)
    new = np.empty_like(state)
    new[:,0,:] = gate[0,0]*state[:,0,:] + gate[0,1]*state[:,1,:]
    new[:,1,:] = gate[1,0]*state[:,0,:] + gate[1,1]*state[:,1,:]
    return new.ravel()


def cnot(state, c, t):
    n = len(state)
    new = state.copy()
    idx = np.arange(n)

    mask = ((idx>>c)&1)==1
    i0 = idx[mask & (((idx>>t)&1)==0)]
    i1 = i0 ^ (1<<t)

    new[i0], new[i1] = state[i1], state[i0]
    return new


# ─────────────────────────────────────────────
# IMPROVED ANSATZ
# ─────────────────────────────────────────────
def ansatz(params, n_qubits, depth):

    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0

    k = 0
    for d in range(depth):

        # rotation layer
        for q in range(n_qubits):

            theta = params[k]; k+=1
            phi   = params[k]; k+=1

            ry = np.array([
                [np.cos(theta/2), -np.sin(theta/2)],
                [np.sin(theta/2),  np.cos(theta/2)]
            ])

            rz = np.array([
                [np.exp(-1j*phi/2), 0],
                [0, np.exp(1j*phi/2)]
            ])

            state = apply_gate(state, q, ry)
            state = apply_gate(state, q, rz)

        # ring entanglement
        for q in range(n_qubits):
            state = cnot(state, q, (q+1)%n_qubits)

    return state


# ─────────────────────────────────────────────
# ENERGY
# ─────────────────────────────────────────────
def energy(state, H):
    return np.real(np.vdot(state, H @ state))


# ─────────────────────────────────────────────
# VQE
# ─────────────────────────────────────────────
def run_vqe(H, n_qubits, depth=4, restarts=8):

    Hs = get_sparse_operator(H, n_qubits)
    n_params = n_qubits * depth * 2

    best = 1e9

    print("\n[VQE] Starting...")

    for r in range(restarts):

        init = 0.1 * np.random.randn(n_params)

        def cost(p):
            s = ansatz(p, n_qubits, depth)
            return energy(s, Hs)

        res = minimize(
            cost,
            init,
            method='Powell',
            options={'maxiter':4000}
        )

        print(f"[VQE] Restart {r+1}: {res.fun:.6f}")

        if res.fun < best:
            best = res.fun

    return best


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("xyz")
    parser.add_argument("--active", type=int, default=6)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=8)

    args = parser.parse_args()

    atoms = parse_xyz(args.xyz)

    H, mol, n_qubits = build_active_hamiltonian(
        atoms, args.active
    )

    print("\nRunning VQE...")
    vqe = run_vqe(H, n_qubits, args.depth, args.restarts)

    print("\n===== RESULTS =====")
    print("HF Energy :", mol.hf_energy)
    print("VQE Energy:", vqe)
    print("Error     :", abs(vqe - mol.hf_energy))


if __name__ == "__main__":
    main()