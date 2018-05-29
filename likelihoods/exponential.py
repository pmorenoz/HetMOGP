# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import sys
import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from GPy.util.misc import safe_exp, safe_square
from scipy.misc import logsumexp

class Exponential(Likelihood):
    """
    Exponential likelihood with a latent function over its parameter

    """

    def __init__(self, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(Exponential, self).__init__(gp_link, name='Exponential')

    def pdf(self, f, y, Y_metadata=None):
        b = safe_exp(-f)
        pdf = safe_exp(-y/b) / b
        return pdf

    def logpdf(self, f, y, Y_metadata=None):
        b = safe_exp(-f) # a veces se va a zero -> np.clip
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        logpdf = - np.log(b) - (y/b)
        return logpdf

    def mean(self, f, Y_metadata=None):
        b = safe_exp(-f)
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        mean = b
        return mean

    def mean_sq(self, f, Y_metadata=None):
        b = safe_exp(-f)
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        mean_sq = safe_square(b)
        return mean_sq

    def variance(self, f, Y_metadata=None):
        b = safe_exp(-f)
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        var = safe_square(b)
        return var

    def samples(self, f, num_samples, Y_metadata=None):
        b = safe_exp(-f)
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        samples = np.random.exponential(scale=b)
        return samples

    def dlogp_df(self, f, y, Y_metadata=None):
        b = safe_exp(-f)
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        dlogp = 1 - (y/b)
        return dlogp

    def d2logp_df2(self, f, y, Y_metadata=None):
        b = safe_exp(-f)
        b = np.clip(b, 1e-9, 1e9)  # numerical stability
        d2logp = - y/b
        return d2logp

    def var_exp(self, Y, m, v, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points

        gh_w = gh_w / np.sqrt(np.pi)
        m, v, Y = m.flatten(), v.flatten(), Y.flatten()
        f = gh_f[None, :] * np.sqrt(2. * v[:, None]) + m[:, None]
        logp = self.logpdf(f, np.tile(Y[:, None], (1, f.shape[1])))
        var_exp = logp.dot(gh_w[:, None])
        return var_exp

    def var_exp_derivatives(self, Y, m, v, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points
        gh_w = gh_w / np.sqrt(np.pi)
        m, v, Y = m.flatten(), v.flatten(), Y.flatten()
        f = gh_f[None, :] * np.sqrt(2. * v[:, None]) + m[:, None]
        dlogp_df = self.dlogp_df(f, np.tile(Y[:, None], (1, f.shape[1])))
        d2logp_df2 = self.d2logp_df2(f, np.tile(Y[:, None], (1, f.shape[1])))
        var_exp_dm = dlogp_df.dot(gh_w[:, None])
        var_exp_dv = 0.5 * d2logp_df2.dot(gh_w[:, None])
        return var_exp_dm, var_exp_dv

    def predictive(self, m, v, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points()
        else:
            gh_f, gh_w = gh_points

        gh_w = gh_w / np.sqrt(np.pi)
        m, v = m.flatten(), v.flatten()
        f = gh_f[None, :] * np.sqrt(2. * v[:, None]) + m[:, None]
        mean = self.mean(f)
        var = self.variance(f).dot(gh_w[:, None]) + self.mean_sq(f).dot(gh_w[:, None]) - np.square(
            mean.dot(gh_w[:, None]))
        mean_pred = mean.dot(gh_w[:, None])
        var_pred = var
        return mean_pred, var_pred

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, num_samples, D))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d][:, None]
            var_fd_star = v_F_star[:, d][:, None]
            F_samples[:,:,d] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(self.logpdf(F_samples[:,:,0], Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        log_predictive = (1/num_samples)*log_pred.sum()

        return log_predictive

    def get_metadata(self):
        dim_y = 1
        dim_f = 1
        dim_p = 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return False