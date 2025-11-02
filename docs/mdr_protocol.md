# Minimum Data Requirement (MDR) Protocol 
ECHO est  activado.
MissclimatePy implements a local-only imputation framework based on spatial (x, y, z) and calendar (t) features. 
The MDR protocol determines the minimum configuration of neighbors (K), observations per station, and inclusion rate needed to achieve stable imputation quality. 
ECHO est  activado.
**Key concepts:** 
- *Inclusion rate:* Fraction of valid training data used from neighbor stations. 
- *MDR:* Smallest configuration (K, min_obs, train_frac) satisfying accuracy thresholds (RMSE, Rý). 
- *Synthetic masking:* Randomly hides fractions of known values to assess reconstruction accuracy. 
ECHO est  activado.
The protocol ensures that local models remain robust under different data availability conditions without requiring external covariates. 
