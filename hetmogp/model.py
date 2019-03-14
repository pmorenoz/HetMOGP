# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import numpy as np
import matplotlib.pyplot as plt

import GPy
from GPy.core.parameterization.param import Param
from GPy.util import choleskies

from hetmogp.inference import Inference
from hetmogp import util
from hetmogp import multi_output
from hetmogp.util import draw_mini_slices

class HetMOGP(GPy.core.SparseGP):
    """
    Description:   Model class for Heterogeneous  Multi-output Gaussian processes
    Class methods:
                        * -- initialization + elbo + gradients -- *
                        1. __init__
                        2. log_likelihood (elbo)
                        3. parameters_changes

                        * -- stochastic optimization -- *
                        4. set_data
                        5. new_batch
                        6. stochastic_grad
                        7. callback

                        * -- prediction -- *
                        8. predictive

    """

##   1.   ##############################################################################################################

    def __init__(self, X, Y, Z, kern_list, likelihood, Y_metadata, name='HetMOGP', batch_size=None):
        """
        :param X:           Input data
        :param Y:           (Heterogeneous) Output data
        :param Z:           Inducing inputs
        :param kern_list:   Kernel functions of GP priors
        :param likelihood:  (Heterogeneous) Likelihoods
        :param Y_metadata:  Linking info between F->likelihoods
        :param name:        Model name
        :param batch_size:  Size of batch for stochastic optimization

        Description: Initialization method for the model class
        """

        #---------------------------------------#     INITIALIZATIONS     #--------------------------------------------#
        #######   Initialization of class variables  #######
        self.batch_size = batch_size
        self.kern_list = kern_list
        self.likelihood = likelihood
        self.Y_metadata = Y_metadata

        #######   Heterogeneous Data  #######
        self.Xmulti = X
        self.Ymulti = Y

        #######  Batches of Data for Stochastic Mode   #######
        self.Xmulti_all, self.Ymulti_all = X, Y
        if batch_size is None:
            self.stochastic = False
            Xmulti_batch, Ymulti_batch = X, Y
        else:
            #######   Makes a climin slicer to make drawing minibatches much quicker   #######
            self.stochastic = True
            self.slicer_list = []
            [self.slicer_list.append(draw_mini_slices(Xmulti_task.shape[0], self.batch_size)) for Xmulti_task in self.Xmulti]
            Xmulti_batch, Ymulti_batch = self.new_batch()
            self.Xmulti, self.Ymulti = Xmulti_batch, Ymulti_batch

        #######   Model dimensions {M, Q, D}  #######
        self.num_inducing = Z.shape[0]  # M
        self.num_latent_funcs = len(kern_list)  # Q
        self.num_output_funcs = likelihood.num_output_functions(self.Y_metadata)

        ####### Inducing points Z #######
        self.Xdim = Z.shape[1]
        Z = np.tile(Z, (1, self.num_latent_funcs))

        #######   Inference   #######
        inference_method = Inference()

        #######  Model class (and inherited classes) super-initialization  #######
        super(HetMOGP, self).__init__(X=Xmulti_batch[0][1:10], Y=Ymulti_batch[0][1:10], Z=Z, kernel=kern_list[0],
                                     likelihood=likelihood, mean_function=None, X_variance=None, inference_method=inference_method,
                                     Y_metadata=Y_metadata, name=name, normalizer=False)

        #######  Initialization of the Multi-output GP mixing  #######
        self.W_list, self.kappa_list = multi_output.random_W_kappas(self.num_latent_funcs, self.num_output_funcs, rank=1)
        _, self.B_list = multi_output.LCM(input_dim=self.Xdim, output_dim=self.num_output_funcs, rank=1, kernels_list=self.kern_list,
                                  W_list=self.W_list, kappa_list=self.kappa_list)

        ####### Initialization of Variational Parameters (q_u_means = \mu, q_u_chols = lower_triang(S))  #######
        self.q_u_means = Param('m_u', 0 * np.random.randn(self.num_inducing, self.num_latent_funcs) +
                               0 * np.tile(np.random.randn(1, self.num_latent_funcs), (self.num_inducing, 1)))
        chols = choleskies.triang_to_flat(np.tile(np.eye(self.num_inducing)[None, :, :], (self.num_latent_funcs, 1, 1)))
        self.q_u_chols = Param('L_u', chols)

        #-----------------------------#   LINKS FOR OPTIMIZABLE PARAMETERS     #---------------------------------------#

        ####### Linking and Un-linking of parameters and hyperaparameters (for ParamZ optimizer)  #######
        self.unlink_parameter(self.kern)  # Unlink SparseGP default param kernel
        self.link_parameter(self.Z, index=0)
        self.link_parameter(self.q_u_means)
        self.link_parameters(self.q_u_chols)
        [self.link_parameter(kern_q) for kern_q in kern_list]  # link all kernels
        [self.link_parameter(B_q) for B_q in self.B_list]

        ####### EXTRA. Auxiliary variables  #######
        self.vem_step = True  # [True=VE-step, False=VM-step]
        self.ve_count = 0
        self.elbo = np.zeros((1, 1))

##   2.   ##############################################################################################################

    def log_likelihood(self):
        """
        Description: Returns the lower bound on the log-marginal likelihood of the heterogeneous MOGP model. (ELBO)
        """
        return self._log_marginal_likelihood

##   3.   ##############################################################################################################

    def parameters_changed(self):
        """
        Description: Updates the "object.gradient" attribute of parameter variables for being used by the optimizer. In
        other words, loads derivatives of the ELBO wrt. variational, hyper- and linear combination parameters into the
        model, taken these ones from the inference class [see inference.py -> gradients()].
        """
        ####### Dimensions #######
        D = self.likelihood.num_output_functions(self.Y_metadata)
        N = self.X.shape[0]
        M = self.num_inducing
        T = len(self.likelihood.likelihoods_list)
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()

        ####### Batch Scaling (Stochastic VI) #######
        self.batch_scale = []
        [self.batch_scale.append(float(self.Xmulti_all[t].shape[0])/float(self.Xmulti[t].shape[0])) for t in range(T)]

        # -------------------------------#   ELBO + BASIC GRADIENTS (Chain Rule)    #----------------------------------#
        self._log_marginal_likelihood, self.gradients = self.inference_method.variational_inference(q_u_means=self.q_u_means,
                                                                        q_u_chols=self.q_u_chols, X=self.Xmulti, Y=self.Ymulti, Z=self.Z,
                                                                        kern_list=self.kern_list, likelihood=self.likelihood,
                                                                        B_list=self.B_list, Y_metadata=self.Y_metadata, batch_scale=self.batch_scale)

        #------------------------------------#   ALL GRADIENTS UPDATE     #--------------------------------------------#
        Z_grad = np.zeros_like(self.Z.values)
        for q, kern_q in enumerate(self.kern_list):

            #-----------------------------#   GRADIENTS OF VARIATIONAL PARAMETERS   #----------------------------------#
            #######  Update gradients of variational parameter  #######
            self.q_u_means[:, q:q + 1].gradient = self.gradients['dL_dmu_u'][q]
            self.q_u_chols[:, q:q + 1].gradient = self.gradients['dL_dL_u'][q]

            # ----------------------.-------#   GRADIENTS OF HYPERPARAMETERS    #--------------------------------------#
            #######   Update gradients of kernel hyperparameters: lengthscale and variance   #######
            kern_q.update_gradients_full(self.gradients['dL_dKmm'][q], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim])
            grad = kern_q.gradient.copy()

            #######  Update gradients of (multi-output) kernel hyperparameters: W + kappa   #######
            Kffdiag = []
            KuqF = []
            for d in range(D):
                #######  main correction consisted of building Kffdiag by multiplying also kern_q.Kdiag  #######
                Kffdiag.append(kern_q.Kdiag(self.Xmulti[f_index[d]]) * self.gradients['dL_dKdiag'][q][d])
                KuqF.append(kern_q.K(self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]]) * self.gradients['dL_dKmn'][q][d])

            util.update_gradients_diag(self.B_list[q], Kffdiag)
            Bgrad = self.B_list[q].gradient.copy()
            util.update_gradients_Kmn(self.B_list[q], KuqF, D)
            Bgrad += self.B_list[q].gradient.copy()
            self.B_list[q].gradient = Bgrad

            #######   Re-update gradients of kernel hyperparameters: lengthscale and variance (second term) #######
            for d in range(self.likelihood.num_output_functions(self.Y_metadata)):
                kern_q.update_gradients_full(self.B_list[q].W[d] * self.gradients['dL_dKmn'][q][d],self.Z[:, q * self.Xdim:q * self.Xdim + self.Xdim],self.Xmulti[f_index[d]])
                grad += kern_q.gradient.copy()
                kern_q.update_gradients_diag(self.B_list[q].B[d,d] *self.gradients['dL_dKdiag'][q][d], self.Xmulti[f_index[d]])
                grad += kern_q.gradient.copy()  # Juan J. wrote this line

            kern_q.gradient = grad

            #######  Update gradients of inducing points #######
            if not self.Z.is_fixed:
                Z_grad[:,q*self.Xdim:q*self.Xdim+self.Xdim] += kern_q.gradients_X(self.gradients['dL_dKmm'][q], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim]).copy()
                for d in range(self.likelihood.num_output_functions(self.Y_metadata)):
                    Z_grad[:,q*self.Xdim:q*self.Xdim+self.Xdim]+= self.B_list[q].W[d]*kern_q.gradients_X(self.gradients['dL_dKmn'][q][d], self.Z[:, q * self.Xdim:q * self.Xdim + self.Xdim],self.Xmulti[f_index[d]]).copy()

        self.Z.gradient[:] = Z_grad

##   4.   ##############################################################################################################

    def set_data(self, X, Y):
        """
        Description: Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients
        """
        self.Xmulti, self.Ymulti = X, Y

##   5.   ##############################################################################################################

    def new_batch(self):
        """
        Return a new batch of X and Y by taking a chunk of data from the complete X and Y
        """
        T = len(self.likelihood.likelihoods_list)
        Xmulti_batch = []
        Ymulti_batch = []
        for t in range(T):
            i_task = next(self.slicer_list[t])
            Xmulti_batch.append(self.Xmulti_all[t][i_task])
            Ymulti_batch.append(self.Ymulti_all[t][i_task])
        return Xmulti_batch, Ymulti_batch

##   6.   ##############################################################################################################

    def stochastic_grad(self, parameters):
        """
        Description: Return the stochastic gradients using the new batch
        """
        self.set_data(*self.new_batch())
        stochastic_gradients = self._grads(parameters)
        return stochastic_gradients

##   7.   ##############################################################################################################

    def callback(self, i, max_iter, verbose=True, verbose_plot=False):
        """
        Description: Printing function for the optimizer. Shows progress of the cost function.
        """
        ll = self.log_likelihood()
        self.elbo[i['n_iter'],0] =  self.log_likelihood()[0]
        if verbose:
            if i['n_iter']%50 ==0:
                print('svi - iteration '+str(i['n_iter'])+'/'+str(int(max_iter)))

        if verbose_plot:
            plt.ion()
            plt.show()
            plt.plot(i['n_iter'],ll,'k+')
            plt.draw()
            plt.pause(1e-5)

        if i['n_iter'] > max_iter:
            return True
        return False

##   8.   ##############################################################################################################

    def predictive(self, Xnew, output_function_ind=None, kern_list=None):
        """
        Description: Make a prediction for the latent output function values over Xnew.
        """
        f_ind = self.Y_metadata['function_index'].flatten()
        if output_function_ind is None:
            output_function_ind = 0
        d = output_function_ind
        if kern_list is None:
            kern_list = self.kern_list

        Xmulti_all_new = self.Xmulti_all.copy()
        Xmulti_all_new[f_ind[d]] = Xnew

        posteriors = self.inference_method.posteriors(q_u_means=self.q_u_means, q_u_chols=self.q_u_chols, X=Xmulti_all_new,
                                                      Y=self.Ymulti_all, Z=self.Z, kern_list=self.kern_list, likelihood=self.likelihood,
                                                      B_list=self.B_list, Y_metadata=self.Y_metadata)

        posterior_output = posteriors[output_function_ind]
        Kx = np.zeros((Xmulti_all_new[f_ind[d]].shape[0], Xnew.shape[0]))
        Kxx = np.zeros((Xnew.shape[0], Xnew.shape[0]))
        for q, B_q in enumerate(self.B_list):
            Kx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(Xmulti_all_new[f_ind[d]], Xnew)
            Kxx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(Xnew, Xnew)

        mu = np.dot(Kx.T, posterior_output.woodbury_vector)
        Kxx = np.diag(Kxx)
        var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior_output.woodbury_inv).T, Kx) * Kx[None, :, :], 1)).T

        return mu, np.abs(var)  # corregir

########################################################################################################################