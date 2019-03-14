# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

from GPy import kern
from GPy.util import linalg
import warnings
import numpy as np

########################################################################################################################

"""
Multi-output Gaussian Process priors

Methods:    * --- mixing model --- *
            1. ICM / Intinsic Coregionalization Model
            2. LCM / Linear Coregionalization Model
            
            * --- matrices --- *
            3. cross_covariance
            4. function_covariance
            5. latent_funs_cov
            
            * --- independent GP prior --- *
            6. latent_functions_prior / now RBF
            
            * --- initialization --- *
            7. random_W_kappas    
"""

##   1.   ##############################################################################################################

def ICM(input_dim, output_dim, kernel, rank, W=None, kappa=None, name='ICM'):
    """
    Description: Builds a kernel for an Intrinsic Coregionalization Model
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

##   2.   ##############################################################################################################

def LCM(input_dim, output_dim, kernels_list, W_list, kappa_list, rank, name='B_q'):
    """
    Description: Builds a kernel for an Linear Coregionalization Model
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

##   3.   ##############################################################################################################

def cross_covariance(X, Z, B, kernel_list, d):
    """
    Description: Builds the cross-covariance cov[f_d(x),u(z)] of a Multi-output GP
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
    return Kfdu

##   4.   ##############################################################################################################

def function_covariance(X, B, kernel_list, d):
    """
    Description: Builds the cross-covariance Kfdfd = cov[f_d(x),f_d(x)] of a Multi-output GP
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

##   5.   ##############################################################################################################

def latent_funs_cov(Z, kernel_list):
    """
    Description: Builds the full-covariance cov[u(z),u(z)] of a Multi-output GP for a Sparse approximation
    :param Z: Inducing Points
    :param kernel_list: Kernels of u_q functions priors
    :return: Kuu
    """
    Q = len(kernel_list)
    M,Dz = Z.shape
    Xdim = int(Dz/Q)
    Kuu = np.empty((Q, M, M))
    Luu = np.empty((Q, M, M))
    Kuui = np.empty((Q, M, M))
    for q, kern in enumerate(kernel_list):
        Kuu[q, :, :] = kern.K(Z[:,q*Xdim:q*Xdim+Xdim],Z[:,q*Xdim:q*Xdim+Xdim])
        Luu[q, :, :] = linalg.jitchol(Kuu[q, :, :])
        Kuui[q, :, :], _ = linalg.dpotri(np.asfortranarray(Luu[q, :, :]))
    return Kuu, Luu, Kuui

##   6.   ##############################################################################################################

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
        #######   RBF GP prior  #######
        kern_q = kern.RBF(input_dim=input_dim, lengthscale=lenghtscale[q], variance=variance[q], name='rbf')# \
        kern_q.name = 'kern_q'+str(q)
        kern_list.append(kern_q)
    return kern_list

##   7.   ##############################################################################################################

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

########################################################################################################################