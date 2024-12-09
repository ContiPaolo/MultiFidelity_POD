# Multi-fidelity reduced-order surrogate modeling
Source code of the paper [*Multi-fidelity reduced-order surrogate modeling.*](http://arxiv.org/abs/2309.00325).

![featured](https://github.com/ContiPaolo/MultiFidelity_POD/assets/51111500/152482e1-f75c-4115-b49c-d72ed0f2be72)

## Dataset:
Full datasets are available at https://doi.org/10.5281/zenodo.8316324

## Tutorial:
Tutorial for multi-fidelity regression and multi-fidelity reduced-order modeling of Burger's equation.
[Tutorial (solution)](https://colab.research.google.com/github/ContiPaolo/Multifidelity-Tutorial/blob/main/Tutorial_Multi_Fidelity_(solution).ipynb#scrollTo=JwVPQmFEpVkc)

## Abstract:

High-fidelity numerical simulations of partial differential equations (PDEs) given a restricted computational budget can significantly limit the number of parameter configurations considered and/or time window eval- uated for modeling a given system. Multi-fidelity surrogate modeling aims to leverage less accurate, lower- fidelity models that are computationally inexpensive in order to enhance predictive accuracy when high- fidelity data are limited or scarce. However, low-fidelity models, while often displaying important qualitative spatio-temporal features, fail to accurately capture the onset of instability and critical transients observed in the high-fidelity models, making them impractical as surrogate models. To address this shortcoming, we present a new data-driven strategy that combines dimensionality reduction with multi-fidelity neural network surrogates. The key idea is to generate a spatial basis by applying the classical proper orthogonal decom- position (POD) to high-fidelity solution snapshots, and approximate the dynamics of the reduced states - time-parameter-dependent expansion coefficients of the POD basis - using a multi-fidelity long-short term memory (LSTM) network. By mapping low-fidelity reduced states to their high-fidelity counterpart, the proposed reduced-order surrogate model enables the efficient recovery of full solution fields over time and parameter variations in a non-intrusive manner. The generality and robustness of this method is demon- strated by a collection of parametrized, time-dependent PDE problems where the low-fidelity model can be defined by coarser meshes and/or time stepping, as well as by misspecified physical features. Importantly, the onset of instabilities and transients are well captured by this surrogate modeling technique.

