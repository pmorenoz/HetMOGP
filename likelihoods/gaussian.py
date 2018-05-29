# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from scipy.stats import norm
from scipy.misc import logsumexp


class Gaussian(Likelihood):
    """
    Gaussian likelihood with a latent function over its mean parameter

    """

    def __init__(self, sigma=None, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        if sigma is None:
            self.sigma = 0.5
        else:
            self.sigma = sigma

        super(Gaussian, self).__init__(gp_link, name='Gaussian')

    def pdf(self, f, y, Y_metadata=None):
        pdf = norm.pdf(y, loc=f)
        return pdf

    def logpdf(self, f, y, Y_metadata=None):
        logpdf = norm.logpdf(y, loc=f)
        return logpdf

    def samples(self, f , num_samples, Y_metadata=None):
        samples = np.random.normal(loc=f, scale=self.sigma)
        #samples = np.random.normal(loc=f, size=(num_samples, f.shape[1]))
        return samples

    def var_exp(self, Y, m, v, gh_points=None, Y_metadata=None):
        # Variational Expectation (Analytical)
        # E_q(fid)[log(p(yi|fid))]
        lik_v = np.square(self.sigma)
        m, v, Y = m.flatten(), v.flatten(), Y.flatten()
        m = m[:,None]
        v = v[:,None]
        Y = Y[:,None]
        var_exp = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(lik_v) \
                  - 0.5 * (np.square(Y) + np.square(m) + v - (2 * m * Y)) / lik_v
        return var_exp

    def var_exp_derivatives(self, Y, m, v, gh_points=None, Y_metadata=None):
        # Variational Expectations of derivatives
        lik_v = np.square(self.sigma)
        m, v, Y = m.flatten(), v.flatten(), Y.flatten()
        m = m[:,None]
        v = v[:,None]
        Y = Y[:,None]
        var_exp_dm = - (m - Y) / lik_v
        var_exp_dv = - 0.5 * (1 / np.tile(lik_v, (m.shape[0],1)))
        return var_exp_dm, var_exp_dv

    def predictive(self, m, v, Y_metadata):
        mean_pred = m
        var_pred = np.square(self.sigma) + v
        return mean_pred, var_pred

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, num_samples, D))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d][:, None]
            var_fd_star = v_F_star[:, d][:, None]
            F_samples[:, :, d] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

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