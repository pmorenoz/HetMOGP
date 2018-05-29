# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from scipy.stats import multinomial
from functools import reduce
from GPy.util.misc import safe_exp, safe_square
from scipy.misc import logsumexp


class Categorical(Likelihood):
    """
    Categorical likelihood with K dimensional
    Needs (K-1) latent functions (see Link Functions)

    """
    def __init__(self, K, gp_link=None):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(Categorical, self).__init__(gp_link, name='Categorical')
        self.K = K

    def pdf(self, F, y, Y_metadata=None):
        Y_oneK = self.onehot(y)
        eF = safe_exp(F)
        den = 1 + eF.sum(1)[:, None]
        p = eF / np.tile(den, eF.shape[1])
        p = np.hstack((p, 1 / den))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        p = p / np.tile(p.sum(1)[:,None], (1, p.shape[1]))
        pdf = multinomial.pmf(x=Y_oneK, n=1, p=p)
        return pdf

    def logpdf(self, F, y, Y_metadata=None):
        Y_oneK = self.onehot(y)
        eF = safe_exp(F)
        den = 1 + eF.sum(1)[:, None]
        p = eF / np.tile(den, eF.shape[1])
        p = np.hstack((p, 1 / den))
        p = np.clip(p, 1e-9, 1- 1e-9)
        p = p / np.tile(p.sum(1)[:,None], (1, p.shape[1]))
        logpdf = multinomial.logpmf(x=Y_oneK, n=1, p=p)
        return logpdf

    def logpdf_sampling(self, F, y, Y_metadata=None):
        Y_oneK = self.onehot(y)
        eF = safe_exp(F)
        den = 1 + eF.sum(1)[:, None, :]
        p = eF / np.tile(den, (1, eF.shape[1] ,1))
        p = np.hstack((p, 1 / den))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        p = p / np.tile(p.sum(1)[:,None,:], (1, p.shape[1],1))
        Y_oneK_rep = np.tile(Y_oneK, (eF.shape[2],1))
        p_rep = np.empty((p.shape[0]*p.shape[2],p.shape[1]))
        for s in range(p.shape[2]):
            p_rep[s * p.shape[0]:(s * p.shape[0]) + p.shape[0], :] = p[:, :, s]

        logpdf = multinomial.logpmf(x=Y_oneK_rep, n=1, p=p_rep)
        logpdf = logpdf.reshape(p.shape[0], p.shape[2])
        return logpdf

    def samples(self, F, num_samples,Y_metadata=None):
        eF = safe_exp(F)
        den = 1 + eF.sum(1)[:, None]
        p = eF / np.tile(den, eF.shape[1])
        p = np.hstack((p, 1 / den))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        p = p / np.tile(p.sum(1)[:,None], (1, p.shape[1]))
        samples = np.empty((F.shape[0], self.K))
        for i in range(F.shape[0]):
            samples[i,:] = multinomial.rvs(n=1, p=p[i,:], size=1)
        return self.invonehot(Y=samples)

    def onehot(self, y):
        # One-Hot Encoding of Categorical Data
        Y_onehot = np.zeros((y.shape[0], self.K))
        for k in range(self.K):
            Y_onehot[:,k,None] = (y==k+1).astype(np.int)
        return Y_onehot

    def invonehot(self, Y):
        # One-Hot Encoding of Categorical Data
        ycat = np.where( Y == 1)[1] + 1
        return ycat[:,None]

    def rho_k(self, F, k):
        # Probability of class k: P(y=k)
        Kminus1 = F.shape[1]
        eF = safe_exp(F)
        rho = eF / (1 + np.tile(eF.sum(1)[:,None], (1, F.shape[1])))
        rho = np.clip(rho, 1e-9, 1. - 1e-9)  # numerical stability
        rho = rho / np.tile(rho.sum(1)[:,None], (1, rho.shape[1]))
        if k>Kminus1:
            rho_k = 1 - rho.sum(1)
        else:
            rho_k = rho[:,k]
        return rho_k

    def dlogp_df(self, df, F, y, Y_metadata=None):
        # df: indicated the derivated function f from F
        Y_oneK = self.onehot(y)
        eF = safe_exp(F)
        den = 1 + eF.sum(1)[:, None]
        p = eF[:, df, None] / den
        p = np.clip(p, 1e-9, 1. - 1e-9)  # numerical stability
        p = p / np.tile(p.sum(1)[:,None], (1, p.shape[1]))
        yp = Y_oneK*np.tile(p, (1, Y_oneK.shape[1])) #old, new is simpler
        dlogp = Y_oneK[:,df,None] - yp.sum(1)[:,None] #old, new is simpler
        #dlogp = Y_oneK[:,df,None] - p
        return dlogp

    def d2logp_df2(self, df, F, y, Y_metadata=None):
        # df: indicated the derivated function f from F
        Y_oneK = self.onehot(y)
        eF = safe_exp(F)
        den = 1 + eF.sum(1)[:, None]
        num = F + np.tile(F[:,df,None],(1,F.shape[1]))
        enum = safe_exp(num)
        enum[:,df] = safe_exp(F[:,df])
        num = enum.sum(1)[:,None]
        p = num / safe_square(den) #añadir clip
        #p = p / np.tile(p.sum(1), (1, p.shape[1]))
        yp = Y_oneK*np.tile(p, (1, Y_oneK.shape[1])) #old, new is simpler
        d2logp =  - yp.sum(1)[:,None] #old, new is simpler
        return d2logp

    def var_exp(self, y, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=10)
        else:
            gh_f, gh_w = gh_points
        D = M.shape[1]
        # grid-size and fd tuples
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1]*(D+1)
            expanded_fd_tuple[d+1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1]*(D+1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple),D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d])*np.sqrt(2*np.reshape(V[:,d],mv_tuple)) \
                    + np.reshape(M[:,d],mv_tuple)
            F[:,d,None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')

        # function evaluation
        Y_full = np.repeat(y, gh_f.shape[0]**D, axis=0)
        logp = self.logpdf(F, Y_full)
        logp = logp.reshape(tuple(grid_tuple))

        # calculating quadrature
        var_exp = logp.dot(gh_w) / np.sqrt(np.pi)
        for d in range(D-1):
            var_exp = var_exp.dot(gh_w) / np.sqrt(np.pi)

        return var_exp[:,None]

    def var_exp_derivatives(self, y, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=10)
        else:
            gh_f, gh_w = gh_points
        N = M.shape[0]
        D = M.shape[1]
        # grid-size and fd tuples
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(2 * np.reshape(V[:, d], mv_tuple)) \
                    + np.reshape(M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')

        # function evaluation
        Y_full = np.repeat(y, gh_f.shape[0] ** D, axis=0)
        var_exp_dm = np.empty((N,D))
        var_exp_dv = np.empty((N,D))
        for d in range(D):
            # wrt to the mean
            dlogp = self.dlogp_df(d, F, Y_full)
            dlogp = dlogp.reshape(tuple(grid_tuple))
            ve_dm = dlogp.dot(gh_w) / np.sqrt(np.pi)
            # wrt to the variance
            d2logp = self.d2logp_df2(d, F, Y_full)
            d2logp = d2logp.reshape(tuple(grid_tuple))
            ve_dv = d2logp.dot(gh_w) / np.sqrt(np.pi)
            for fd in range(D - 1):
                ve_dm = ve_dm.dot(gh_w) / np.sqrt(np.pi)
                ve_dv = ve_dv.dot(gh_w) / np.sqrt(np.pi)

            var_exp_dm[:,d] = ve_dm
            var_exp_dv[:,d] = 0.5 * ve_dv
        return var_exp_dm, var_exp_dv

    def predictive(self, M, V, gh_points=None, Y_metadata=None):
        # Variational Expectation
        # gh: Gaussian-Hermite quadrature
        if gh_points is None:
            gh_f, gh_w = self._gh_points(T=10)
        else:
            gh_f, gh_w = gh_points
        N = M.shape[0]
        D = M.shape[1]
        # grid-size and fd tuples
        expanded_F_tuples = []
        grid_tuple = [M.shape[0]]
        for d in range(D):
            grid_tuple.append(gh_f.shape[0])
            expanded_fd_tuple = [1] * (D + 1)
            expanded_fd_tuple[d + 1] = gh_f.shape[0]
            expanded_F_tuples.append(tuple(expanded_fd_tuple))

        # mean-variance tuple
        mv_tuple = [1] * (D + 1)
        mv_tuple[0] = M.shape[0]
        mv_tuple = tuple(mv_tuple)

        # building, normalizing and reshaping the grids
        F = np.zeros((reduce(lambda x, y: x * y, grid_tuple), D))
        for d in range(D):
            fd = np.zeros(tuple(grid_tuple))
            fd[:] = np.reshape(gh_f, expanded_F_tuples[d]) * np.sqrt(2 * np.reshape(V[:, d], mv_tuple)) \
                    + np.reshape(M[:, d], mv_tuple)
            F[:, d, None] = fd.reshape(reduce(lambda x, y: x * y, grid_tuple), -1, order='C')

        # function evaluation
        mean_pred = np.empty((N, D))
        var_pred = np.zeros((N, D))
        for d in range(D):
            # wrt to the mean
            mean_k = self.rho_k(F, d)
            mean_k = mean_k.reshape(tuple(grid_tuple))
            mean_pred_k = mean_k.dot(gh_w) / np.sqrt(np.pi)
            # wrt to the variance
            # NOT IMPLEMENTED
            for fd in range(D - 1):
                mean_pred_k = mean_pred_k.dot(gh_w) / np.sqrt(np.pi)

            mean_pred[:, d] = mean_pred_k
        return mean_pred, var_pred

    def log_predictive(self, Ytest, mu_F_star, v_F_star, num_samples):
        Ntest, D = mu_F_star.shape
        F_samples = np.empty((Ntest, D, num_samples))
        # function samples:
        for d in range(D):
            mu_fd_star = mu_F_star[:, d, None]
            var_fd_star = v_F_star[:, d, None]
            F_samples[:, d, :] = np.random.normal(mu_fd_star, np.sqrt(var_fd_star), size=(Ntest, num_samples))

        # monte-carlo:
        log_pred = -np.log(num_samples) + logsumexp(self.logpdf_sampling(F_samples, Ytest), axis=-1)
        log_pred = np.array(log_pred).reshape(*Ytest.shape)
        log_predictive = (1/num_samples)*log_pred.sum()

        return log_predictive

    def get_metadata(self):
        dim_y = 1
        dim_f = self.K - 1
        dim_p = self.K - 1
        return dim_y, dim_f, dim_p

    def ismulti(self):
        # Returns if the distribution is multivariate
        return True