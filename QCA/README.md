# Semi-Quantum Lenia

Here we store all the functions used for semi quantum lenia.
It can be used for both conway's original game of life and lenia.

    - Qcell: convert from NxNx1 arrays to NxNx2 quantum state array by applying normalization to each element
    - neighbouring_sites/liveliness: find the Moore's neighbourhood sum
    - SQGOL: semi-quantum game of life quantum version of GoL
    - cont_SQGOL: lenia version of quantum update (continuous)
    - plot_update_Qrule: plot growth functions for visualization

Note: Functions like cont_SQGOL_with_phase, updating the cells with interference pattern, have not been used.
These utilities needs further development