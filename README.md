# BandedPlusSpiked

This repository contains code to reproduce the results from the paper:  

**“Approximate MLE of High-Dimensional STAP Covariance Matrices with Banded & Spiked Structure — A Convex Relaxation Approach.”**

---

## Reproducing Figures

- **Different Number of Pulses**  
  Code: `BandedplusSpikedComputation.m`  
  Dataset: 
  -`Cofar_Monte100_ICM_14_31_Pulses_32.mat` (for 32 pulses)
  -`Cofar_Monte100_ICM_14_31.mat` (for 64 pulses)
- **Regularization Parameter (α) Variation**  
  Code: `Lambda_variation.m`  
  Dataset: `Cofar.mat`

- **PRF Variation**  
  Code: `PRFVariation.m`  
  Datasets:  
  - `Cofar_PRF_1100.mat`  
  - `Cofar_PRF_1650.mat`  
  - `Cofar_PRF_2200.mat`

- **Figure 5 (ε-plots)**  
  Code: `EpsilonPlot.m`

---

## Dataset Generation

- `Cofar.mat`, `Cofar_PRF_1100.mat`, `Cofar_PRF_1650.mat`, `Cofar_PRF_2200.mat`  
  Generated using **`cofar_example_v2_1.m`**.  
  > *Contact co-author **Sandeep Gogineni** for access to RFView modules required to run this code.*

- `Cofar_Monte100_ICM_14_31_Pulses_32.mat`, `Cofar_Monte100_ICM_14_31.mat`  
  Generated using **`cofar_example_v2_MonteCarlo.m`**.  
  > *Contact co-author **Sandeep Gogineni** for access to RFView modules required to run this code.*

---

## Note
The RFView® modules required for dataset generation are proprietary. If you need access, please reach out to the co-author listed above.
