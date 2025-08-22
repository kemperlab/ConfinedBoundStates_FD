# BoundStateSpectrum_FunctionalDerivatives

This code implements the functional derivatives process implemented in the paper linked below to compute the bound state spectrum of a 1-D mixed-field Ising model exhibiting confined dynamics.

_Probing Bound State Relaxation Dynamics in Systems Out-of-Equilibrium on Quantum Computers_

Authors: Heba A. Labib, Goksu Can Toga, J. K. Freericks, A. F. Kemper

ArXiv: [https://arxiv.org/abs/2507.22988](https://arxiv.org/abs/2507.22988)

Abstract: Pump-probe spectroscopy is a powerful tool for probing response dynamics of quantum many-body systems in and out-of-equilibrium. Quantum computers have proved useful in simulating such experiments by exciting the system, evolving, and then measuring observables to first order, all in one setting. Here, we use this approach to investigate the mixed-field Ising model, where the longitudinal field plays the role of a confining potential that prohibits the spread of the excitations, spinons, or domain walls into space. We study the discrete bound states that arise from such a setting and their evolution under different quench dynamics by initially pumping the chain out of equilibrium and then probing various non-equal time correlation functions. Finally, we study false vacuum decay, where initially one expects unhindered propagation of the ground state, or true vacuum, bubbles into the lattice, but instead sees the emergence of Bloch oscillations that are directly the reason for the long-lived oscillations in this finite-size model. Our work sets the stage for simulating systems out-of-equilibrium on classical and quantum computers using pump-probe experiments without needing ancillary qubits.

___

## Contents

* fd_conf_ising.py $\rightarrow$ Data Generation for both the response functions for a $H = H_0+\phi(t) Z_i$ and the computation of the various correlation functions using functional derivatives.
* Two classes:

  spin_models.py $\rightarrow$ this class includes many spin Hamiltonians relevant to similar non-equilibrium calculations where the mixed-field ising model is chosen for the purposes of our study. 
  
  probes_f.py $\rightarrow$ this class includes different pulse shapes $\phi(t)$ that couple to the operator we measure to obtain the corresponding Green functions with appropriate time and frequency resolution.
* fd_plots.ipynb $\rightarrow$ Data processing and figures-generating Jupyter Notebook 






