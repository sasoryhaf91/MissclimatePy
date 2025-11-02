# Minimum Data Requirement (MDR) Protocol 

MissclimatePy implements a local-only imputation framework based on spatial (x, y, z) and calendar (t) features. 
The MDR protocol determines the minimum configuration of neighbors (K), observations per station, and inclusion rate needed to achieve stable imputation quality. 

**Key concepts:** 
- *Inclusion rate:* Fraction of valid training data used from neighbor stations. 
- *MDR:* Smallest configuration (K, min_obs, train_frac) satisfying accuracy thresholds (RMSE, R^2). 
- *Synthetic masking:* Randomly hides fractions of known values to assess reconstruction accuracy. 

The protocol ensures that local models remain robust under different data availability conditions without requiring external covariates. 
