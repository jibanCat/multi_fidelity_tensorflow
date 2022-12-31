# A collection of practice notebooks using TFP's Gaussian process

I saw a great example in [tensorflow probability](https://www.tensorflow.org/probability/examples/Gaussian_Process_Regression_In_TFP) on using Gaussian process, including assigning priors to GP hyperparameters, marginal likelihood estimation, and Hamiltonian Monte Carlo sampling of the hyperparameters.
So I tried to modify it to my problems.

Below are some of my example notebooks for practicing. Not all of them are highly annotated. But if there's any question, please let me know.

1. `01_Pk_Emulator`: An example of using TFP GPs on power spectrum emulation. The major difference here is we train a batch of independent GPs. So we can predict vectorized output. The Automatic Relevance Determination is borrowed from TFP issue 248, but the current suggested way is probably using `FeatureScaled` kernel.
2. `02_Lya_Emulator`: (working on it).
3. `03_Multi-Fidelity_Emulator`: An example of using TFP GPs to build a linear multi-fidelity GP (AR1). The major modification is adding a learnable scale parameter between two GPs. The toy example data is borrowed from [`emukit`](https://nbviewer.org/github/EmuKit/emukit/blob/main/notebooks/Emukit-tutorial-multi-fidelity.ipynb). In contradict to the emukit's notebook, I added learnable noises to the GPs, which I found it's more stable than without noises.
4. `04_GP_with_Correlated_Bins_for_Vector_Statistics`: An example of using TFP GP to interpolate the vector output in example 01. In example 01, we predict vector P(k) by building num(k) independent GPs. In example 04, we build a kernel = K(x, x')K(k, k') to interpolate the k, so we only need one GP.

