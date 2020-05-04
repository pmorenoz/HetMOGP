# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield


from GPy import kern
from GPy.util import linalg
import random
import warnings
import numpy as np
import climin
from functools import partial
import matplotlib.pyplot as plt


def  get_batch_scales(X_all, X):
    batch_scales = []
    for t, X_all_task in enumerate(X_all):
        batch_scales.append(float(X_all_task.shape[0]) / float(X[t].shape[0]))
    return batch_scales

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

def mini_slices(n_samples, batch_size):
    """Yield slices of size `batch_size` that work with a container of length
    `n_samples`."""
    n_batches, rest = divmod(n_samples, batch_size)
    if rest != 0:
        n_batches += 1

    return [slice(i * batch_size, (i + 1) * batch_size) for i in range(n_batches)]


def draw_mini_slices(n_samples, batch_size, with_replacement=False):
    slices = mini_slices(n_samples, batch_size)
    idxs = list(range(len(slices)))  # change this line

    if with_replacement:
        yield random.choice(slices)
    else:
        while True:
            random.shuffle(list(idxs))
            for i in idxs:
                yield slices[i]


def latent_functions_prior(Q, lenghtscale=None, variance=None, input_dim=None):
    if lenghtscale is None:
        lenghtscale = np.random.rand(Q)
    else:
        lenghtscale = lenghtscale

    if variance is None:
        variance = np.random.rand(Q)
    else:
        variance = variance
    kern_list = []
    for q in range(Q):
        kern_q = kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')# \
        kern_q.name = 'kern_q'+str(q)
        kern_list.append(kern_q)
    return kern_list

def random_W_kappas(Q,D,rank, experiment=False):
    W_list = []
    kappa_list = []
    for q in range(Q):
        p = np.random.binomial(n=1, p=0.5*np.ones((D,1)))
        Ws = p*np.random.normal(loc=0.5, scale=0.5, size=(D,1)) - (p-1)*np.random.normal(loc=-0.5, scale=0.5, size=(D,1))
        W_list.append(Ws / np.sqrt(rank)) # deberían ser tanto positivos como negativos
        if experiment:
            kappa_list.append(np.zeros(D))
        else:
            kappa_list.append(np.zeros(D))
    return W_list, kappa_list


def ICM(input_dim, output_dim, kernel, rank, W=None, kappa=None, name='ICM'):
    """
    Builds a kernel for an Intrinsic Coregionalization Model
    :input_dim: Input dimensionality (does not include dimension of indices)
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    kern_q = kernel.copy()
    if kernel.input_dim != input_dim:
        kernel.input_dim = input_dim
        warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")
    B = kern.Coregionalize(input_dim=input_dim, output_dim=output_dim, rank=rank, W=W, kappa=kappa)
    B.name = name
    K = kern_q.prod(B, name=name)
    return K, B


def LCM(input_dim, output_dim, kernels_list, W_list, kappa_list, rank, name='B_q'):
    """
    Builds a kernel for an Linear Coregionalization Model
    :input_dim: Input dimensionality (does not include dimension of indices)
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    B_q = []
    K, B = ICM(input_dim, output_dim, kernels_list[0], W=W_list[0], kappa=kappa_list[0], rank=rank, name='%s%s' %(name,0))
    B_q.append(B)
    for q, kernel in enumerate(kernels_list[1:]):
        Kq, Bq = ICM(input_dim, output_dim, kernel, W=W_list[q+1], kappa=kappa_list[q+1], rank=rank, name='%s%s' %(name,q+1))
        B_q.append(Bq)
        K += Kq
    return K, B_q

def cross_covariance(X, Z, B, kernel_list, d):
    """
    Builds the cross-covariance cov[f_d(x),u(z)] of a Multi-output GP
    :param X: Input data
    :param Z: Inducing Points
    :param B: Coregionalization matric
    :param kernel_list: Kernels of u_q functions
    :param d: output function f_d index
    :return: Kfdu
    """
    N,_ = X.shape
    M,Dz = Z.shape
    Q = len(B)
    Xdim = int(Dz/Q)
    Kfdu = np.empty([N,M*Q])
    for q, B_q in enumerate(B):
        Kfdu[:, q * M:(q * M) + M] = B_q.W[d] * kernel_list[q].K(X, Z[:, q*Xdim:q*Xdim+Xdim])
        #Kfdu[:,q*M:(q*M)+M] = B_q.W[d]*kernel_list[q].K(X,Z[:,q,None])
        #Kfdu[:, q * M:(q * M) + M] = B_q.B[d,d] * kernel_list[q].K(X, Z[:,q,None])
    return Kfdu

def function_covariance(X, B, kernel_list, d):
    """
    Builds the cross-covariance Kfdfd = cov[f_d(x),f_d(x)] of a Multi-output GP
    :param X: Input data
    :param B: Coregionalization matrix
    :param kernel_list: Kernels of u_q functions
    :param d: output function f_d index
    :return: Kfdfd
    """
    N,_ = X.shape
    Kfdfd = np.zeros((N, N))
    for q, B_q in enumerate(B):
        Kfdfd += B_q.B[d,d]*kernel_list[q].K(X,X)
    return Kfdfd

def latent_funs_cov(Z, kernel_list):
    """
    Builds the full-covariance cov[u(z),u(z)] of a Multi-output GP
    for a Sparse approximation
    :param Z: Inducing Points
    :param kernel_list: Kernels of u_q functions priors
    :return: Kuu
    """
    Q = len(kernel_list)
    M,Dz = Z.shape
    Xdim = int(Dz/Q)
    #Kuu = np.zeros([Q*M,Q*M])
    Kuu = np.empty((Q, M, M))
    Luu = np.empty((Q, M, M))
    Kuui = np.empty((Q, M, M))
    for q, kern in enumerate(kernel_list):
        Kuu[q, :, :] = kern.K(Z[:,q*Xdim:q*Xdim+Xdim],Z[:,q*Xdim:q*Xdim+Xdim])
        Luu[q, :, :] = linalg.jitchol(Kuu[q, :, :])
        Kuui[q, :, :], _ = linalg.dpotri(np.asfortranarray(Luu[q, :, :]))
    return Kuu, Luu, Kuui

def generate_toy_U(X,Q):
    arg = np.tile(X, (1,Q))
    rnd = np.tile(np.random.rand(1,Q), (X.shape))
    U = 2*rnd*np.sin(10*rnd*arg + np.random.randn(1)) + 2*rnd*np.cos(20*rnd*arg + np.random.randn(1))
    return U

def _gradient_reduce_numpy(coreg, dL_dK, index, index2):
    index, index2 = index[:,0], index2[:,0]
    dL_dK_small = np.zeros_like(coreg.B)
    for i in range(coreg.output_dim):
        tmp1 = dL_dK[index==i]
        for j in range(coreg.output_dim):
            dL_dK_small[j,i] = tmp1[:,index2==j].sum()
    return dL_dK_small

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

def update_gradients_diag(coreg, dL_dKdiag):
    dL_dKdiag_small = np.array([dL_dKdiag_task.sum() for dL_dKdiag_task in dL_dKdiag])
    coreg.W.gradient = coreg.W*dL_dKdiag_small[:, None] # should it be 2*..?
    coreg.kappa.gradient = dL_dKdiag_small

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

def update_gradients_Kmn(coreg, dL_dK, D):
    dW = np.zeros((D,1))
    dkappa = np.zeros((D)) # not used
    for d in range(D):
        dW[d,:] = dL_dK[d].sum()

    coreg.W.gradient = dW
    coreg.kappa.gradient = dkappa

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

def vem_algorithm(model, stochastic=False, vem_iters=None, step_rate=None ,verbose=False, optZ=True, verbose_plot=False, non_chained=True):
    model['.*.lengthscale'].fix()
    if vem_iters is None:
        vem_iters = 5

    model['.*.kappa'].fix() # must be always fixed
    model.elbo = np.empty((vem_iters,1))

    if stochastic is False:

        for i in range(vem_iters):
            # VARIATIONAL E-STEP
            model['.*.lengthscale'].fix()
            model['.*.variance'].fix()
            model.Z.fix()
            model['.*.W'].fix()

            #model.q_u_means.fix()
            #model.q_u_chols.fix()
            model.q_u_means.unfix()
            model.q_u_chols.unfix()
            model.optimize(messages=verbose, max_iters=100)
            print('iteration ('+str(i+1)+') VE step, ELBO='+str(model.log_likelihood().flatten()))

            # VARIATIONAL M-STEP
            model['.*.lengthscale'].unfix()
            model['.*.variance'].unfix()
            if optZ:
                model.Z.unfix()
            if non_chained:
                model['.*.W'].unfix()

            model.q_u_means.fix()
            model.q_u_chols.fix()
            model.optimize(messages=verbose, max_iters=100)
            print('iteration (' + str(i+1) + ') VM step, ELBO=' + str(model.log_likelihood().flatten()))

    else:
            if step_rate is None:
                step_rate = 0.01

            sto_iters = vem_iters
            model.elbo = np.empty((sto_iters+1,1))
            optimizer = climin.Adadelta(model.optimizer_array, model.stochastic_grad, step_rate=step_rate, momentum=0.9)
            c_full = partial(model.callback, max_iter=sto_iters, verbose=verbose, verbose_plot=verbose_plot)
            optimizer.minimize_until(c_full)

    return model
