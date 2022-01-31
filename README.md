# HGCN

The official repo of "HGCN: Harmonic Gated Compensation Network For Speech Enhancement", which was accepted at ICASSP2022.

## How to use

#### step1: Calculate and test the harmonic integral matrix
```shell
cd harmonic_intefral
python make_integral_matrix.py
# Integral matrix (harmonic_integrate_matrix.npy, U in our paper) and harmonic locations (harmonic_loc.npy, Harmonic locations corresponding to each candidate pitch) will be generated in the dir. 
```

#### step2: Prepare the label of speech energy detector
```shell
cd speech_energy_detector_label
python mean_threshold.py
```
