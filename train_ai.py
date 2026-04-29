import numpy as np

# =========================================
# DATA (FULL DATASET)
# =========================================
# Format:
# [n_qubits, n_active_orbitals, (VQE - HF)]

X = np.array([
    [4, 2, -0.0205],        # H2
    [10, 6, 0.6052],        # H2O
    [12, 6, 0.7307],        # NH3
    [12, 6, 0.7316],        # CH4 (unseen earlier)
    [12, 6, 0.4462],        # C2H6 (unseen earlier)
    [12, 6, 0.7253],        # CH3OH (unseen earlier)
    [12, 6, 0.3490580141364603],   # CO2  (VQE - HF)
    [12, 6, 0.40702994074925414],  # N2
    [12, 6, 0.3827086536308144],   # HCN
])

# =========================================
# TARGET (CASCI - VQE)
# =========================================
y = np.array([
    -0.0205,   # H2
    -0.6319,   # H2O
    -0.7715,   # NH3
    -0.7738,   # CH4
    -0.4615,   # C2H6
    -0.7394,   # CH3OH
    -0.3490580141364603,   # CO2
    -0.40702994074925414,  # N2
    -0.3827086536308144,   # HCN
])

# =========================================
# FEATURE ENGINEERING
# =========================================
def build_features(X):
    q = X[:, 0]
    a = X[:, 1]
    dE = X[:, 2]

    return np.column_stack([
        dE,           # MOST IMPORTANT
        dE**2,
        q,
        a,
        q * dE,
        a * dE
    ])

X_feat = build_features(X)

# =========================================
# NORMALIZATION
# =========================================
mean = X_feat.mean(axis=0)
std = X_feat.std(axis=0)
std[std == 0] = 1

X_norm = (X_feat - mean) / std

# =========================================
# ADD BIAS
# =========================================
X_norm = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])

# =========================================
# RIDGE REGRESSION
# =========================================
lam = 0.01
I = np.eye(X_norm.shape[1])

w = np.linalg.inv(X_norm.T @ X_norm + lam * I) @ X_norm.T @ y

print("\nLearned weights:\n", w)

# =========================================
# SAVE MODEL
# =========================================
np.save("weights.npy", w)
np.save("mean.npy", mean)
np.save("std.npy", std)

print("\nModel saved: weights.npy, mean.npy, std.npy")

# =========================================
# PREDICTION FUNCTION
# =========================================
def predict(n_qubits, n_active, vqe_energy, hf_energy):
    dE = vqe_energy - hf_energy
    x = np.array([[n_qubits, n_active, dE]])

    x_feat = build_features(x)
    x_norm = (x_feat - mean) / std
    x_norm = np.hstack([x_norm, [[1]]])

    return (x_norm @ w)[0]

# =========================================
# TEST (NH3 + CH4 + CH3OH)
# =========================================
print("\n===== TESTING MODEL =====")

# NH3
hf = -55.45408420040727
vqe = -54.72339823222654
corr = predict(12, 6, vqe, hf)
print("\nNH3 Correction:", corr)
print("NH3 Final:", vqe + corr)

# CH4
hf = -39.726715311543025
vqe = -38.99507580093866
corr = predict(12, 6, vqe, hf)
print("\nCH4 Correction:", corr)
print("CH4 Final:", vqe + corr)

# CH3OH
hf = -113.51367092561713
vqe = -112.78837194658814
corr = predict(12, 6, vqe, hf)
print("\nCH3OH Correction:", corr)
print("CH3OH Final:", vqe + corr)