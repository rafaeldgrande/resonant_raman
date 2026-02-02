# resonant_raman
Codes to calculate resonant raman based on excited state forces (exciton-phonon coefficients)

Step 1: Calculate excited state forces 
Step 2: Convert forces from cartesian to phonon basis using the cart2ph_eigvec.py code from excited_state_forces repository
Step 3: Assemble all exciton-phonon coefficients into a single .h5 file using the assemble_exciton_phonon_coeffs.py script
Step 4: Run resonant_raman.py to get the resonant raman spectrum
