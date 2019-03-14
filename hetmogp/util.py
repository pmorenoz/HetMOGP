# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import random
import numpy as np
import climin
from functools import partial

########################################################################################################################

"""
Util methods for HetMOGP module

Methods:    * --- stochastic optimization --- *
            1. get_batch_scales
            2. mini_slices
            3. draw_mini_slices

            * --- MOGP kernel gradients --- *
            4. _gradient_reduce_numpy
            5. _gradient_B
            6. update_gradients_diag
            7. update_gradients_full
            8. update_gradients_Kmn
            9. gradients_coreg
            10. gradients_coreg_diag

            * --- algorithm --- *
            11. vem_algorithm
            
            * --- toy data generation --- *
            12. true_u_functions
            13. true_f_functions
            14. generate_toy_U  
"""


##   1.   ##############################################################################################################

def  get_batch_scales(X_all, X):
    """
    Description: Returns proportions of batches w.r.t. the total number of samples for the stochastic gradient scaling.
    """
    batch_scales = []
    for t, X_all_task in enumerate(X_all):
        batch_scales.append(float(X_all_task.shape[0]) / float(X[t].shape[0]))
    return batch_scales

##   2.   ##############################################################################################################

def mini_slices(n_samples, batch_size):
    """
    Description: Yield slices of size `batch_size` that work with a container of length 'n_samples'.
    """
    n_batches, rest = divmod(n_samples, batch_size)
    if rest != 0:
        n_batches += 1

    return [slice(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]

##   3.   ##############################################################################################################

def draw_mini_slices(n_samples, batch_size, with_replacement=False):
    """
    Description: Returns indexes of the samples in the new batch of data (here called slices)
    """
    slices = mini_slices(n_samples, batch_size)
    idxs = list(range(len(slices)))  # change this line

    if with_replacement:
        yield random.choice(slices)
    else:
        while True:
            random.shuffle(list(idxs))
            for i in idxs:
                yield slices[i]

##   4.   ##############################################################################################################

def _gradient_reduce_numpy(coreg, dL_dK, index, index2):
    index, index2 = index[:,0], index2[:,0]
    dL_dK_small = np.zeros_like(coreg.B)
    for i in range(coreg.output_dim):
        tmp1 = dL_dK[index==i]
        for j in range(coreg.output_dim):
            dL_dK_small[j,i] = tmp1[:,index2==j].sum()
    return dL_dK_small

##   5.   ##############################################################################################################

def _gradient_B(coreg, dL_dK, index, index2):
    index, index2 = index[:,0], index2[:,0]
    B = coreg.B
    isqrtB = 1 / np.sqrt(B)
    dL_dK_small = np.zeros_like(B)
    for i in range(coreg.output_dim):
        tmp1 = dL_dK[index==i]
        for j in range(coreg.output_dim):
            dL_dK_small[j,i] = (0.5 * isqrtB[i,j] * tmp1[:,index2==j]).sum()
    return dL_dK_small

##   6.   ##############################################################################################################

def update_gradients_diag(coreg, dL_dKdiag):
    dL_dKdiag_small = np.array([dL_dKdiag_task.sum() for dL_dKdiag_task in dL_dKdiag])
    coreg.W.gradient = 2.*coreg.W*dL_dKdiag_small[:, None] # should it be 2*..? R/Yes Pablo, it should be :)
    coreg.kappa.gradient = dL_dKdiag_small

##   7.   ##############################################################################################################

def update_gradients_full(coreg, dL_dK, X, X2=None):
    index = np.asarray(X, dtype=np.int)
    if X2 is None:
        index2 = index
    else:
        index2 = np.asarray(X2, dtype=np.int)

    dL_dK_small = _gradient_reduce_numpy(coreg, dL_dK, index, index2)
    dkappa = np.diag(dL_dK_small).copy()
    dL_dK_small += dL_dK_small.T
    dW = (coreg.W[:, None, :]*dL_dK_small[:, :, None]).sum(0)

    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

##   8.   ##############################################################################################################

def update_gradients_Kmn(coreg, dL_dK, D):
    dW = np.zeros((D,1))
    dkappa = np.zeros((D)) # not used
    for d in range(D):
        dW[d,:] = dL_dK[d].sum()

    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

##   9.   ##############################################################################################################

def gradients_coreg(coreg, dL_dK, X, X2=None):
    index = np.asarray(X, dtype=np.int)
    if X2 is None:
        index2 = index
    else:
        index2 = np.asarray(X2, dtype=np.int)

    dK_dB = _gradient_B(coreg, dL_dK, index, index2)
    dkappa = np.diag(dK_dB).copy()
    dK_dB += dK_dB.T
    dW = (coreg.W[:, None, :]*dK_dB[:, :, None]).sum(0)
    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

##   10.   ##############################################################################################################

def gradients_coreg_diag(coreg, dL_dKdiag, kern_q, X, X2=None):
    # dL_dKdiag is (NxD)
    if X2 is None:
        X2 = X
    N,D =  dL_dKdiag.shape
    matrix_sum = np.zeros((D,1))
    for d in range(D):
        matrix_sum[d,0] = np.sum(np.diag(kern_q.K(X, X2)) * dL_dKdiag[:,d,None])

    dW = 2 * coreg.W * matrix_sum
    dkappa = matrix_sum
    return dW, dkappa

##   11.   ##############################################################################################################

def vem_algorithm(model, vem_iters=None, maxIter_perVEM = None, step_rate=None ,verbose=False, optZ=True, verbose_plot=False, non_chained=True):
    if vem_iters is None:
        vem_iters = 5
    if maxIter_perVEM is None:
        maxIter_perVEM = 100

    model['.*.kappa'].fix() # must be always fixed!
    if model.batch_size is None:

        for i in range(vem_iters):
            #######  VARIATIONAL E-STEP  #######
            model['.*.lengthscale'].fix()
            model['.*.variance'].fix()
            model.Z.fix()
            model['.*.W'].fix()

            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            model.optimize(messages=verbose, max_iters=maxIter_perVEM)
            print('iteration ('+str(i+1)+') VE step, log_likelihood='+str(model.log_likelihood().flatten()))

            #######  VARIATIONAL M-STEP   #######
            model['.*.lengthscale'].unfix()
            model['.*.variance'].unfix()
            if optZ:
                model.Z.unfix()
            if non_chained:
                model['.*.W'].unfix()

            model.q_u_means.fix()
            model.q_u_chols.fix()
            model.optimize(messages=verbose, max_iters=maxIter_perVEM)
            print('iteration (' + str(i+1) + ') VM step, log_likelihood=' + str(model.log_likelihood().flatten()))

    else:

        if step_rate is None:
            step_rate = 0.01

        model.elbo = np.empty((maxIter_perVEM*vem_iters+2, 1))
        model.elbo[0,0]=model.log_likelihood()
        c_full = partial(model.callback, max_iter=maxIter_perVEM, verbose=verbose, verbose_plot=verbose_plot)

        for i in range(vem_iters):
            #######  VARIATIONAL E-STEP  #######
            model['.*.lengthscale'].fix()
            model['.*.variance'].fix()
            model.Z.fix()
            model['.*.W'].fix()

            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            optimizer = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=step_rate,decay_mom1=1 - 0.9, decay_mom2=1 - 0.999)
            optimizer.minimize_until(c_full)
            print('iteration (' + str(i + 1) + ') VE step, mini-batch log_likelihood=' + str(model.log_likelihood().flatten()))

            #######  VARIATIONAL M-STEP  #######
            model['.*.lengthscale'].unfix()
            model['.*.variance'].unfix()
            if optZ:
                model.Z.unfix()
            if non_chained:
                model['.*.W'].unfix()

            model.q_u_means.fix()
            model.q_u_chols.fix()
            optimizer = climin.Adam(model.optimizer_array, model.stochastic_grad, step_rate=step_rate,decay_mom1=1 - 0.9, decay_mom2=1 - 0.999)
            optimizer.minimize_until(c_full)
            print('iteration (' + str(i + 1) + ') VM step, mini-batch log_likelihood=' + str(model.log_likelihood().flatten()))

    return model

##   12.   ##############################################################################################################

def true_u_functions(X_list, Q):
    u_functions = []
    amplitude = (1.5-0.5)*np.random.rand(Q,3) + 0.5
    freq = (3-1)*np.random.rand(Q,3) + 1
    shift = 2*np.random.rand(Q,3)
    for X in X_list:
        u_task = np.empty((X.shape[0],Q))
        for q in range(Q):
            u_task[:,q,None] = 3*amplitude[q,0]*np.cos(freq[q,0]*np.pi*X + shift[q,0]*np.pi) - \
                               2*amplitude[q,1]*np.sin(2*freq[q,1]*np.pi*X + shift[q,1]*np.pi) + \
                               amplitude[q,2] * np.cos(4*freq[q, 2] * np.pi * X + shift[q, 2] * np.pi)

        u_functions.append(u_task)
    return u_functions

##   13.   ##############################################################################################################

def true_f_functions(true_u, W_list, D, likelihood_list, Y_metadata):
    true_f = []
    f_index = Y_metadata['function_index'].flatten()
    d_index = Y_metadata['d_index'].flatten()
    for t, u_task in enumerate(true_u):
        Ntask = u_task.shape[0]
        _, num_f_task, _ = likelihood_list[t].get_metadata()
        F = np.zeros((Ntask, num_f_task))
        for q, W in enumerate(W_list):
            for d in range(D):
                if f_index[d] == t:
                    F[:,d_index[d],None] += np.tile(W[d].T, (Ntask, 1)) * u_task[:, q, None]

        true_f.append(F)
    return true_f

##   14.   ##############################################################################################################

def generate_toy_U(X,Q):
    arg = np.tile(X, (1,Q))
    rnd = np.tile(np.random.rand(1,Q), (X.shape))
    U = 2*rnd*np.sin(10*rnd*arg + np.random.randn(1)) + 2*rnd*np.cos(20*rnd*arg + np.random.randn(1))
    return U

########################################################################################################################