
import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.pyplot as plt

h5_file = 'exciton_phonon_couplings.h5'
eigvals_file = 'eigenvalues_b1.dat'
Emax = 12
dE = 0.1
Ex = np.arange(0.0, Emax, dE)  # eV
gamma = 0.05  # eV


print(f'Reading data from {h5_file}')
with h5py.File(h5_file, 'r') as hf:
    rpa_diag_data = hf['rpa_diag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    rpa_offdiag_data = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
print('Data read successfully.')


print(f'Reading exciton energies from {eigvals_file}')
data_eigvals_file = np.loadtxt(eigvals_file)  # shape: (Nexciton, 4)
exc_energies = data_eigvals_file[:, 0]  # in eV
dip_moments = data_eigvals_file[:, 2]  + 1j * data_eigvals_file[:, 3]
pos_operator = 1j * dip_moments / exc_energies  # <0|r|i> = i * <0|p|i> / E_ixc
print('Exciton energies read successfully.')

Nmodes = rpa_diag_data.shape[0]
Nexc = rpa_diag_data.shape[1]

sum_imode_2d = []

plt.figure()
for imode in range(Nmodes):
    raman_spectrum_mode = np.zeros_like(Ex, dtype=complex)
    
    for iexc in range(Nexc):
        partial_term = (np.abs(pos_operator[iexc])**2 * 
                        rpa_offdiag_data[imode, iexc, iexc] /
                       ( exc_energies[iexc] - Ex + 1j*gamma)**2 )
        
        raman_spectrum_mode += partial_term

    sum_imode_2d.append(-raman_spectrum_mode)


    plt.plot(Ex, np.abs(sum_imode_2d[imode]), label=f'Mode {imode}')
    
sum_imode_3d = []

plt.figure()
for imode in range(Nmodes):
    raman_spectrum_mode = np.zeros_like(Ex, dtype=complex)
    
    for iexc1 in range(Nexc):
        for iexc2 in range(iexc1+1, Nexc):
            partial_term = (pos_operator[iexc1] * pos_operator[iexc2].conjugate() *
                            rpa_offdiag_data[imode, iexc1, iexc2] /
                            ( (exc_energies[iexc1] - Ex + 1j*gamma) *
                                (exc_energies[iexc2] - Ex + 1j*gamma) ) )
            raman_spectrum_mode += partial_term

            partial_term = (pos_operator[iexc2] * pos_operator[iexc1].conjugate() *
                            rpa_offdiag_data[imode, iexc2, iexc1] /
                            ( (exc_energies[iexc2] - Ex + 1j*gamma) *
                                (exc_energies[iexc1] - Ex + 1j*gamma) ) )
            raman_spectrum_mode += partial_term

    sum_imode_3d.append(-raman_spectrum_mode)


    plt.plot(Ex, np.abs(sum_imode_3d[imode]), label=f'Mode {imode} (3d)')


plt.xlabel('Excitation energy (eV)')
plt.ylabel('Raman Intensity (arb. units)')
plt.show()

# # Pre-compute upper triangular indices
# iexc1_idx, iexc2_idx = np.triu_indices(Nexc_plot, k=1)

# # Vectorized computation for all modes and exciton pairs
# energy_diffs = exc_energies[iexc2_idx] - exc_energies[iexc1_idx]
# rpa_offdiag_terms = np.abs(rpa_offdiag_data[:, iexc1_idx, iexc2_idx]).ravel()
# rpa_diag_terms = np.abs(rpa_diag_data[:, iexc1_idx, iexc2_idx]).ravel()
# energy_diffs_all = np.tile(energy_diffs, Nmodes)

