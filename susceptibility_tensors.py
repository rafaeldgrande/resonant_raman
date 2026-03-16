
import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', default='exciton_phonon_couplings.h5')
parser.add_argument('--eigvals_file', default='eigenvalues_b1.dat')
parser.add_argument('--dip_mom_file_b1', default='eigenvalues_b1.dat')
parser.add_argument('--dip_mom_file_b2', default='eigenvalues_b2.dat')
parser.add_argument('--dip_mom_file_b3', default='eigenvalues_b3.dat')
parser.add_argument('--Emax', type=float, default=12)
parser.add_argument('--dE', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.05)
args = parser.parse_args()

h5_file = args.h5_file
eigvals_file = args.eigvals_file
dip_mom_file_b1 = args.dip_mom_file_b1
dip_mom_file_b2 = args.dip_mom_file_b2
dip_mom_file_b3 = args.dip_mom_file_b3
Emax = args.Emax
dE = args.dE
Ex = np.arange(0.0, Emax, dE)  # eV
gamma = args.gamma  # eV


print(f'Reading data from {h5_file}')
with h5py.File(h5_file, 'r') as hf:
    # rpa_diag_data = hf['rpa_diag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    rpa_offdiag_data = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
print('Data read successfully.')


print(f'Reading exciton energies from {eigvals_file}')
data_eigvals_file = np.loadtxt(eigvals_file)  # shape: (Nexciton, 4)
exc_energies = data_eigvals_file[:, 0]  # in eV

data_dip_mom_b1 = np.loadtxt(dip_mom_file_b1)  # shape: (Nexciton, 4)
data_dip_mom_b2 = np.loadtxt(dip_mom_file_b2)  # shape: (Nexciton, 4)
data_dip_mom_b3 = np.loadtxt(dip_mom_file_b3)  # shape: (Nexciton, 4)
dip_moments_b1 = data_dip_mom_b1[:, 2]  + 1j * data_dip_mom_b1[:, 3]
dip_moments_b2 = data_dip_mom_b2[:, 2]  + 1j * data_dip_mom_b2[:, 3]
dip_moments_b3 = data_dip_mom_b3[:, 2]  + 1j * data_dip_mom_b3[:, 3]
pos_operator_b1 = 1j * dip_moments_b1 / exc_energies  # <0|r|i> = i * <0|p|i> / E_ixc
pos_operator_b2 = 1j * dip_moments_b2 / exc_energies
pos_operator_b3 = 1j * dip_moments_b3 / exc_energies

pos_operator_list = [pos_operator_b1, pos_operator_b2, pos_operator_b3]

print('Exciton energies read successfully.')

Nmodes = rpa_offdiag_data.shape[0]
Nexc = rpa_offdiag_data.shape[1]


alpha_tensor_d2 = np.zeros((3, 3, Nmodes, Ex.shape[0]), dtype=complex)
alpha_tensor_d3 = np.zeros((3, 3, Nmodes, Ex.shape[0]), dtype=complex)


# plt.figure()

for ialpha in range(3):
    pos_operator_alpha = pos_operator_list[ialpha]
    for ibeta in range(3):
        pos_operator_beta = pos_operator_list[ibeta]
        
        for imode in range(Nmodes):
            
            # d2
            raman_spectrum_alpha_beta_mode = np.zeros_like(Ex, dtype=complex)
            for iexc in range(Nexc):
                partial_term = (pos_operator_alpha[iexc] * pos_operator_beta[iexc].conjugate() *
                                rpa_offdiag_data[imode, iexc, iexc] /
                            ( exc_energies[iexc] - Ex + 1j*gamma)**2 )
                raman_spectrum_alpha_beta_mode += partial_term
            alpha_tensor_d2[ialpha, ibeta, imode, :] = -raman_spectrum_alpha_beta_mode
            
            # d3
            raman_spectrum_alpha_beta_mode = np.zeros_like(Ex, dtype=complex)            
            for iexc1 in range(Nexc):
                for iexc2 in range(iexc1+1, Nexc):
                    partial_term = (pos_operator_alpha[iexc1] * pos_operator_beta[iexc2].conjugate() *
                                    rpa_offdiag_data[imode, iexc1, iexc2] /
                                    ( (exc_energies[iexc1] - Ex + 1j*gamma) *
                                        (exc_energies[iexc2] - Ex + 1j*gamma) ) )
                    raman_spectrum_alpha_beta_mode += partial_term

                    partial_term = (pos_operator_alpha[iexc2] * pos_operator_beta[iexc1].conjugate() *
                                    rpa_offdiag_data[imode, iexc2, iexc1] /
                                    ( (exc_energies[iexc2] - Ex + 1j*gamma) *
                                        (exc_energies[iexc1] - Ex + 1j*gamma) ) )
                    raman_spectrum_alpha_beta_mode += partial_term

            alpha_tensor_d3[ialpha, ibeta, imode, :] = -raman_spectrum_alpha_beta_mode

# save results to h5 file
output_h5_file = 'susceptibility_tensors.h5'
with h5py.File(output_h5_file, 'w') as hf:
    hf.create_dataset('alpha_tensor_d2', data=alpha_tensor_d2)
    hf.create_dataset('alpha_tensor_d3', data=alpha_tensor_d3)
