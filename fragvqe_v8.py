#!/usr/bin/env python3
"""
FragVQE v8.0
VQE + AI (trained model) + CASCI reference
"""

import argparse
import numpy as np
from scipy.optimize import minimize

from openfermion import MolecularData
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import run_pyscf

from pyscf import mcscf


# =========================================
# LOAD AI MODEL 🔥
# =========================================
w = np.load("weights.npy")
mean = np.load("mean.npy")
std = np.load("std.npy")


def build_features(X):
    q = X[:, 0]
    a = X[:, 1]
    dE = X[:, 2]

    return np.column_stack([
        dE,
        dE**2,
        q,
        a,
        q * dE,
        a * dE
    ])


def ai_correction(n_qubits, n_active, vqe_energy, hf_energy):
    dE = vqe_energy - hf_energy

    x = np.array([[n_qubits, n_active, dE]])
    x_feat = build_features(x)

    x_norm = (x_feat - mean) / std
    x_norm = np.hstack([x_norm, [[1]]])

    return (x_norm @ w)[0]


# =========================================
# XYZ PARSER
# =========================================
def parse_xyz(file):
    lines = open(file).readlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2:2+n]:
        s,x,y,z = line.split()
        atoms.append((s,float(x),float(y),float(z)))
    return atoms


# =========================================
# ACTIVE SPACE
# =========================================
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

    n_occ = n_elec // 2
    start = max(0, n_occ - active_orbitals//2)
    stop  = min(n_orb, start + active_orbitals)

    active_idx = list(range(start, stop))
    core_idx   = list(range(start))

    print(f"[ACTIVE] Orbitals: {active_idx}")

    hamiltonian = mol.get_molecular_hamiltonian(
        occupied_indices=core_idx,
        active_indices=active_idx
    )

    fermion = get_fermion_operator(hamiltonian)
    qubit = jordan_wigner(fermion)

    n_qubits = 2 * len(active_idx)
    print(f"[INFO] Active qubits: {n_qubits}")

    return qubit, mol, n_qubits, active_idx


# =========================================
# CASCI
# =========================================
def compute_casci(mol, active_idx):
    mf = mol._pyscf_data['scf']

    ncas = len(active_idx)
    nelecas = mol.n_electrons - 2*active_idx[0]

    mc = mcscf.CASCI(mf, ncas, nelecas)
    return mc.kernel()[0]


# =========================================
# GATES + ANSATZ (same as before)
# =========================================
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


def ansatz(params, n_qubits, depth):

    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0

    k = 0
    for d in range(depth):
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

        for q in range(0, n_qubits-1, 2):
            state = cnot(state, q, q+1)

        for q in range(1, n_qubits-1, 2):
            state = cnot(state, q, q+1)

    return state


def run_vqe(H, n_qubits, depth=4, restarts=3):

    Hs = get_sparse_operator(H, n_qubits)
    best = 1e9

    for _ in range(restarts):
        params = 0.01 * np.random.randn(2 * n_qubits * depth)

        def cost(p):
            s = ansatz(p, n_qubits, depth)
            return np.real(np.vdot(s, Hs @ s))

        res = minimize(cost, params, method='Powell', options={'maxiter':2000})

        if res.fun < best:
            best = res.fun

    return best


# =========================================
# MAIN
# =========================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("xyz")
    parser.add_argument("--active", type=int, default=6)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=3)

    args = parser.parse_args()

    atoms = parse_xyz(args.xyz)

    H, mol, n_qubits, active_idx = build_active_hamiltonian(atoms, args.active)

    print("\nRunning VQE...")
    vqe_energy = run_vqe(H, n_qubits, args.depth, args.restarts)

    correction = ai_correction(
        n_qubits,
        len(active_idx),
        vqe_energy,
        mol.hf_energy
    )

    final_energy = vqe_energy + correction

    print("\nComputing CASCI...")
    e_casci = compute_casci(mol, active_idx)

    print("\n===== RESULTS =====")
    print("HF Energy       :", mol.hf_energy)
    print("VQE Energy      :", vqe_energy)
    print("AI Correction   :", correction)
    print("Final Energy    :", final_energy)
    print("CASCI           :", e_casci)

    print("\n===== ERRORS =====")
    print("VQE Error       :", abs(vqe_energy - e_casci))
    print("Final Error     :", abs(final_energy - e_casci))


if __name__ == "__main__":
    main()