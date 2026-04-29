# VQE_AI
The core of this project is the Hybrid FragVQE pipeline, which decomposes complex molecular Hamiltonians into manageable chemical fragments. While each fragment is solved using a hardware-efficient VQE ansatz , an AI post-processing step—trained on primary molecular benchmarks—predicts the energy difference between the VQE result and CASCI.
