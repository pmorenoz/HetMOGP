# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import sys
import numpy as np
import GPy
from GPy.inference.latent_function_inference import LatentFunctionInference
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.util import choleskies
from GPy.util import linalg
from hetmogp import util
from collections import namedtuple
from scipy.linalg.blas import dtrmm
import matplotlib.pyplot as plt
import random

qfd = namedtuple("q_fd", "m_fd v_fd Kfdu Afdu S_fd")
qu = namedtuple("q_U", "mu_u chols_u")
pu = namedtuple("p_U", "Kuu Luu Kuui")

class SVMOGPInf(LatentFunctionInference):

    def inference(self, q_u_means, q_u_chols, X, Y, Z, kern_list, likelihood, B_list, Y_metadata, KL_scale=1.0,
                  batch_scale=None, predictive=False):
        M = Z.shape[0]
        T = len(Y)
        if batch_scale is None:
            batch_scale  = [1.0]*T
        Ntask = []
        [Ntask.append(Y[t].shape[0]) for t in range(T)]
        Q = len(kern_list)
        D = likelihood.num_output_functions(Y_metadata)
        Kuu, Luu, Kuui = util.latent_funs_cov(Z, kern_list)
        p_U = pu(Kuu=Kuu, Luu=Luu, Kuui=Kuui)
        q_U = qu(mu_u=q_u_means, chols_u=q_u_chols)

        # for every latent function f_d calculate q(f_d) and keep it as q(F):
        q_F = []
        posteriors_F = []
        f_index = Y_metadata['function_index'].flatten()
        d_index = Y_metadata['d_index'].flatten()

        for d in range(D):
            Xtask = X[f_index[d]]
            q_fd = self.calculate_q_f(X=Xtask, Z=Z, q_U=q_U, p_U=p_U, kern_list=kern_list, B=B_list,
                                      M=M, N=Xtask.shape[0], Q=Q, D=D, d=d)
            # Posterior objects for output functions (used in prediction)
            posterior_fd = Posterior(mean=q_fd.m_fd.copy(), cov=q_fd.S_fd.copy(),
                                     K=util.function_covariance(X=Xtask, B=B_list, kernel_list=kern_list, d=d),
                                     prior_mean=np.zeros(q_fd.m_fd.shape))
            posteriors_F.append(posterior_fd)
            q_F.append(q_fd)

        mu_F = []
        v_F = []
        for t in range(T):
            mu_F_task = np.empty((X[t].shape[0],1))
            v_F_task = np.empty((X[t].shape[0], 1))
            for d, q_fd in enumerate(q_F):
                if f_index[d] == t:
                    mu_F_task = np.hstack((mu_F_task, q_fd.m_fd))
                    v_F_task = np.hstack((v_F_task, q_fd.v_fd))

            mu_F.append(mu_F_task[:,1:])
            v_F.append(v_F_task[:,1:])

        # posterior_Fnew for predictive
        if predictive:
            return posteriors_F
        # inference for rest of cases
        else:
            # Variational Expectations
            VE = likelihood.var_exp(Y, mu_F, v_F, Y_metadata)
            VE_dm, VE_dv = likelihood.var_exp_derivatives(Y, mu_F, v_F, Y_metadata)
            for t in range(T):
                VE[t] = VE[t]*batch_scale[t]
                VE_dm[t] = VE_dm[t]*batch_scale[t]
                VE_dv[t] = VE_dv[t]*batch_scale[t]

            # KL Divergence
            KL = self.calculate_KL(q_U=q_U, p_U=p_U,M=M, Q=Q)

            # Log Marginal log(p(Y))
            F = 0
            for t in range(T):
                F += VE[t].sum()

            log_marginal = F - KL

            # Gradients and Posteriors
            dL_dmu_u = []
            dL_dL_u = []
            dL_dKmm = []
            dL_dKmn = []
            dL_dKdiag = []
            posteriors = []
            for q in range(Q):
                (dL_dmu_q, dL_dL_q, posterior_q, dL_dKqq, dL_dKdq, dL_dKdiag_q) = self.calculate_gradients(q_U=q_U, p_U=p_U,
                                                            q_F=q_F,VE_dm=VE_dm, VE_dv=VE_dv, Ntask=Ntask, M=M, Q=Q, D=D, f_index=f_index, d_index=d_index, q=q)
                dL_dmu_u.append(dL_dmu_q)
                dL_dL_u.append(dL_dL_q)
                dL_dKmm.append(dL_dKqq)
                dL_dKmn.append(dL_dKdq)
                dL_dKdiag.append(dL_dKdiag_q)
                posteriors.append(posterior_q)

            gradients = {'dL_dmu_u':dL_dmu_u, 'dL_dL_u':dL_dL_u, 'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dL_dKdiag}

            return log_marginal, gradients, posteriors, posteriors_F

    def calculate_gradients(self, q_U, p_U, q_F, VE_dm, VE_dv, Ntask, M, Q, D, f_index, d_index,q):
        """
        Calculates gradients of the Log-marginal distribution p(Y) wrt variational
        parameters mu_q, S_q
        """
        # Algebra for q(u) and p(u):
        m_u = q_U.mu_u.copy()
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())
        S_u = np.empty((Q, M, M))
        [np.dot(L_u[q, :, :], L_u[q, :, :].T, S_u[q, :, :]) for q in range(Q)]
        Kuu = p_U.Kuu.copy()
        Luu = p_U.Luu.copy()
        Kuui = p_U.Kuui.copy()
        S_qi, _ = linalg.dpotri(np.asfortranarray(L_u[q, :, :]))

        if np.any(np.isinf(S_qi)):
            raise ValueError("Sqi: Cholesky representation unstable")

        # KL Terms
        dKL_dmu_q = np.dot(Kuui[q,:,:],m_u[:, q, None])
        dKL_dS_q = 0.5 * (Kuui[q,:,:] - S_qi)
        dKL_dKqq = 0.5 * Kuui[q,:,:] - 0.5 * Kuui[q,:,:].dot(S_u[q, :, :]).dot(Kuui[q,:,:]) \
                   - 0.5 * np.dot(Kuui[q,:,:],np.dot(m_u[:, q, None],m_u[:, q, None].T)).dot(Kuui[q,:,:].T)

        # VE Terms
        dVE_dmu_q = np.zeros((M, 1))
        dVE_dS_q = np.zeros((M, M))
        dVE_dKqq = np.zeros((M, M))
        dVE_dKqd = []
        dVE_dKdiag = []

        for d, q_fd in enumerate(q_F):
            Nt = Ntask[f_index[d]]
            dVE_dmu_q += np.dot(q_fd.Afdu[q, :, :].T, VE_dm[f_index[d]][:,d_index[d]])[:, None]
            Adv = q_fd.Afdu[q,:,:].T * VE_dv[f_index[d]][:,d_index[d],None].T
            Adv = np.ascontiguousarray(Adv)
            AdvA = np.dot(Adv.reshape(-1, Nt), q_fd.Afdu[q, :, :]).reshape(M, M)
            dVE_dS_q += AdvA

            # Derivatives dKuquq
            tmp_dv = np.dot(AdvA, S_u[q, :, :]).dot(Kuui[q,:,:])
            dVE_dKqq += AdvA - tmp_dv - tmp_dv.T
            Adm = np.dot(q_fd.Afdu[q, :, :].T, VE_dm[f_index[d]][:,d_index[d],None])
            dVE_dKqq += - np.dot(Adm, np.dot(Kuui[q,:,:], m_u[:, q, None]).T)

            # Derivatives dKuqfd
            tmp = np.dot(S_u[q, :, :], Kuui[q,:,:])
            tmp = 2. * (tmp - np.eye(M))
            dve_kqd = np.dot(np.dot(Kuui[q,:,:], m_u[:, q, None]), VE_dm[f_index[d]][:,d_index[d],None].T)
            dve_kqd += np.dot(tmp.T, Adv)
            dVE_dKqd.append(dve_kqd)

            # Derivatives dKdiag
            dVE_dKdiag.append(VE_dv[f_index[d]][:,d_index[d]])

        dVE_dKqq = 0.5 * (dVE_dKqq + dVE_dKqq.T)
        # Sum of VE and KL terms
        dL_dmu_q = dVE_dmu_q - dKL_dmu_q
        dL_dS_q = dVE_dS_q - dKL_dS_q
        dL_dKqq = dVE_dKqq - dKL_dKqq
        dL_dKdq = dVE_dKqd
        dL_dKdiag = dVE_dKdiag

        # Pass S_q gradients to its low-triangular representation L_q
        chol_u = q_U.chols_u.copy()
        L_q = choleskies.flat_to_triang(chol_u[:,q:q+1])
        dL_dL_q = 2. * np.array([np.dot(a, b) for a, b in zip(dL_dS_q[None,:,:], L_q)])
        dL_dL_q = choleskies.triang_to_flat(dL_dL_q)

        # Posterior
        posterior_q = Posterior(mean=m_u[:, q, None], cov=S_u[q, :, :], K=Kuu[q,:,:], prior_mean=np.zeros(m_u[:, q, None].shape))

        return dL_dmu_q, dL_dL_q, posterior_q, dL_dKqq, dL_dKdq, dL_dKdiag
    

    def calculate_q_f(self, X, Z, q_U, p_U, kern_list, B, M, N, Q, D, d):
        """
        Calculates the mean and variance of q(f_d) as
        Equation: E_q(U)\{p(f_d|U)\}
        """
        # Algebra for q(u):
        m_u = q_U.mu_u.copy()
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())
        S_u = np.empty((Q, M, M))
        [np.dot(L_u[q, :, :], L_u[q, :, :].T, S_u[q, :, :]) for q in range(Q)]

        # Algebra for p(f_d|u):
        Kfdu = util.cross_covariance(X, Z, B, kern_list, d)
        Kuu = p_U.Kuu.copy()
        Luu = p_U.Luu.copy()
        Kuui = p_U.Kuui.copy()
        Kff = util.function_covariance(X, B, kern_list, d)
        Kff_diag = np.diag(Kff)

        # Algebra for q(f_d) = E_{q(u)}[p(f_d|u)]
        Afdu = np.empty((Q, N, M)) #Afdu = K_{fduq}Ki_{uquq}
        m_fd = np.zeros((N, 1))
        v_fd = np.zeros((N, 1))
        S_fd = np.zeros((N, N))
        v_fd += Kff_diag[:,None]
        S_fd += Kff
        for q in range(Q):
            # Expectation part
            R, _ = linalg.dpotrs(np.asfortranarray(Luu[q, :, :]), Kfdu[:, q * M:(q * M) + M].T)
            Afdu[q, :, :] = R.T
            m_fd += np.dot(Afdu[q, :, :], m_u[:, q, None]) #exp
            tmp = dtrmm(alpha=1.0, a=L_u[q, :, :].T, b=R, lower=0, trans_a=0)
            v_fd += np.sum(np.square(tmp), 0)[:,None] - np.sum(R * Kfdu[:, q * M:(q * M) + M].T,0)[:,None] #exp
            S_fd += np.dot(np.dot(R.T,S_u[q, :, :]),R) - np.dot(Kfdu[:, q * M:(q * M) + M],R)

        if (v_fd<0).any():
            print('v negative!')

        q_fd = qfd(m_fd=m_fd, v_fd=v_fd, Kfdu=Kfdu, Afdu=Afdu, S_fd=S_fd)
        return q_fd

    def calculate_KL(self, q_U, p_U, M, Q):
        """
        Calculates the KL divergence (see KL-div for multivariate normals)
        Equation: \sum_Q KL{q(uq)|p(uq)}
        """
        # Algebra for q(u):
        m_u = q_U.mu_u.copy()
        L_u = choleskies.flat_to_triang(q_U.chols_u.copy())
        S_u = np.empty((Q, M, M))
        [np.dot(L_u[q, :, :], L_u[q, :, :].T, S_u[q, :, :]) for q in range(Q)]

        # Algebra for p(u):
        Kuu = p_U.Kuu.copy()
        Luu = p_U.Luu.copy()
        Kuui = p_U.Kuui.copy()

        KL = 0
        for q in range(Q):
            KL += 0.5 * np.sum(Kuui[q, :, :] * S_u[q, :, :]) \
                  + 0.5 * np.dot(m_u[:, q, None].T,np.dot(Kuui[q,:,:],m_u[:, q, None])) \
                  - 0.5 * M \
                  + 0.5 * 2. * np.sum(np.log(np.abs(np.diag(Luu[q, :, :])))) \
                  - 0.5 * 2. * np.sum(np.log(np.abs(np.diag(L_u[q, :, :]))))
        return KL