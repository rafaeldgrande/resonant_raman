
import numpy as np
import matplotlib.pyplot as plt
import h5py

h5_file = 'exciton_phonon_couplings.h5'
with h5py.File(h5_file, 'r') as hf:
    rpa_diag_data = hf['rpa_diag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    rpa_offdiag_data = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    
eigvals_file = 'eigenvalues_b1.dat'
exc_energies = np.loadtxt(eigvals_file)[:, 0]  # shape: (Nexciton,)

Nmodes = rpa_diag_data.shape[0]
Nexc_plot = rpa_diag_data.shape[1]

f = plt.figure(figsize=(10, 6))

# Pre-compute upper triangular indices
iexc1_idx, iexc2_idx = np.triu_indices(Nexc_plot, k=1)

# Vectorized computation for all modes and exciton pairs
energy_diffs = exc_energies[iexc2_idx] - exc_energies[iexc1_idx]
rpa_offdiag_terms = np.abs(rpa_offdiag_data[:, iexc1_idx, iexc2_idx]).ravel()
rpa_diag_terms = np.abs(rpa_diag_data[:, iexc1_idx, iexc2_idx]).ravel()
energy_diffs_all = np.tile(energy_diffs, Nmodes)

# Single scatter call for all points
# plt.scatter(energy_diffs_all, rpa_offdiag_terms, color='red', s=1, alpha=0.5, label='RPA offdiag')
plt.scatter(energy_diffs_all, rpa_diag_terms / energy_diffs_all, color='blue', s=1, alpha=0.5, label='RPA diag')
            
plt.xlabel(r'$\Delta \Omega$ (eV)')
plt.ylabel(r"$|\langle A | dH / dQ | B \rangle| / \Delta \Omega$ (eV/$\rm{\AA^2}$)")
# y axis in log scale
plt.yscale('log')
plt.savefig('exciton_phonon_offdiag_vs_energy_diff.png', dpi=300)
plt.close()
    
