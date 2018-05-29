# Heterogeneous Multi-output Gaussian Processes

This repository contains the implementation of our Heterogeneous Multi-output Gaussian Process model (HetMOGP). It is fully implemented on Python 3.6 and is connected with the GPy package. Our code consists of two main parts: (1) 'hetmogp' module contains model definition, inference, and utilities, (2) 'likelihoods' module is the general library for probability distributions. If you want to include a new likelihood function, please add your new script here. We provide further details about usage below.

Please, if you use this code, cite the following paper:
```
@article{MorenoArtesAlvarez18,
  title={Heterogeneous Multi-output {G}aussian Process Prediction},
  author={Pablo Moreno-Mu\~noz, Antonio Art\'es-Rodríguez and Mauricio A. \'Alvarez},
  journal={arXiv preprint arXiv:1805.07633},
  year={2018}
}
```

## Usage

* Output and input Data (X,Y):

> Y = [Y_hg, Y_ber, Y_cat]

> X = [X_hg, X_ber, X_cat]

* Heterogeneous Likelihood definition:

likelihood_list = [HetGaussian(), Bernoulli(), Categorical(K=3)]

![gap](tmp/gap.png)

![london](tmp/london.png)

## Contributors

Pablo Moreno-Muñoz, Antonio Artés-Rodríguez and Mauricio A. Álvarez

For further information or contact:
```
pmoreno@tsc.uc3m.es
```
