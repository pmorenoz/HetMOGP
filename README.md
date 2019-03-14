# Heterogeneous Multi-output Gaussian Processes

This repository contains the implementation of our Heterogeneous Multi-output Gaussian Process (HetMOGP) model. The entire code is written in Python and connected with the GPy package, specially useful for Gaussian processes. Our code consists of two main blocks:

- **hetmogp**: This block contains all model definitions, inference, and important utilities.
- **likelihoods**: General library of probability distributions for the heterogeneous likelihood construction.

Our tool is a novel extension of multi-output Gaussian processes for handling heterogeneous outputs (from different statistical data-types). The following distributions are already available to be used: [**Gaussian**, **Bernoulli**, **Heteroscedastic Gaussian**, **Categorical**, **Exponential**, **Gamma**, **Beta**, **Poisson**]. We expect to release code for **Student**, **Ordinal**, **Geometric**, **Binomial**, **Multinomial**, **Truncated Gaussian**, **Wishart** and **Dirichlet** likelihood distributions as soon as possible. If you want to contribute or include a new likelihood function, please follow the instructions given below to add your new script to the *likelihoods* module.

Please, if you use this code, cite the following [paper](https://papers.nips.cc/paper/7905-heterogeneous-multi-output-gaussian-process-prediction):
```
@inproceedings{MorenoArtesAlvarez18,
  title =  {Heterogeneous Multi-output {G}aussian Process Prediction},
  author =   {Moreno-Mu\~noz, Pablo and Art\'es-Rodr\'iguez, Antonio and \'Alvarez, Mauricio A},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) 31},
  year =   {2018}
}
```

## Usage

Our Python implementation follows a straightforward sintaxis where you only have to define a list of input and output values, build the heterogeneous likelihood with the desired distributions that you want to predict and call directly to the model class. That is

* Output and input data definition:
```
Y = [Y_real, Y_binary, Y_categorical]
X = [X_real, X_binary, X_categorical]
```
* Heterogeneous Likelihood definition:
```
likelihood_list = [HetGaussian(), Bernoulli(), Categorical(K=3)]
```
* Model and definition:
```
model = HetMOGP(X=X, Y=Y, Z=Z, kern_list=kern_list, likelihood=likelihood, Y_metadata=Y_metadata)
```

A complete example of our model usage can be found in this repository at **notebooks > demo**

## New Likelihoods

The **heterogeneous likehood** structure (based on [Eero Siivola](https://users.aalto.fi/~siivole1/)'s GPy release and [GPstuff](https://github.com/gpstuff-dev/gpstuff)) permits to handle mixed likelihoods with different statistical data types in a very natural way. The idea behind this structure is that any user can add his own distributions easily by following a series of recommendations:

1. Place your **new_distribution.py** under the likelihood directory.
2. Define the **logpdf**, first order **dlogp_df** and second order derivatives **d2logp_df2** of your log-likelihood function.
3. Use **var_exp** and **var_exp_derivatives** for approximating integrals with Gauss-Hermite quadratures.
4. Code your **predictive** and **get_metadata** methods to have all available utilities.

## Examples
* **Missing Gap Prediction:** Predicting in classification problems with information obtained
from parallel regression tasks.
![gap](tmp/gap.png)

* **London House Prices:** Spatial modeling with heterogeneous samples. This is a
demographic example where we mix discrete data (type of house) with real observations
(log-price of house sale contracts).
![london](tmp/london.png)

## Potential Applications
We have collected many ideas about possible applications of our heterogeneous multi-output GP model. 

## Contributors

[Pablo Moreno-Muñoz](http://www.tsc.uc3m.es/~pmoreno/), [Antonio Artés-Rodríguez](http://www.tsc.uc3m.es/~antonio/) and [Mauricio A. Álvarez](https://sites.google.com/site/maalvarezl/)

For further information or contact:
```
pmoreno@tsc.uc3m.es
```
