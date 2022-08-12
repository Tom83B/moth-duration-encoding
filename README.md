# moth-duration-encoding

Supporting code, Jupyter notebooks and data for the manuscript **Stimulus duration encoding occurs early in the moth olfactory pathway** [(Barta et al., 2022)][biorxiv]

The folder `data` contains the used experimental data. In text files are spike trains and stimulus onset / offset timings. `.abf` files contain recordings of the local field potential (LFP) or photo-ionization detector (PID) signal (folders `base recordings` and `PID`).

The associated scripts and Jupyter notebooks are in Python 3.8.5 and the following libraries were used:
* numpy (1.20.3)
* scipy (1.7.1)
* scikit-learn (0.24.2)
* pandas (1.3.3)
* matplotlib (3.4.2)
* neo (0.9.0) - for reading `.abf` files

Jupyter notebooks reproduce the figures in the manuscript. Python scripts provide helpful functions used throughout the Jupyter notebooks and generate simulation and analysis data, which are saved in `data/generated`

[biorxiv]: https://www.biorxiv.org/content/10.1101/2022.07.21.501055v1
