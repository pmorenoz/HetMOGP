# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import numpy as np
import GPy
from hetmogp.util import draw_mini_slices
from hetmogp.svmogp_inf import SVMOGPInf
from GPy.core.parameterization.param import Param
from GPy.plotting.matplot_dep.util import fixed_inputs
import matplotlib.pyplot as plt
from GPy.util import choleskies
from GPy.util.misc import kmm_init
from hetmogp import util
import random

class SVMOGP(GPy.core.SparseGP):
    def __init__(self, X, Y, Z, kern_list, likelihood, Y_metadata, name='SVMOGP', batch_size=None, W_list=None):

        self.batch_size = batch_size
        self.kern_list = kern_list
        self.likelihood = likelihood
        self.Y_metadata = Y_metadata

        self.num_inducing = Z.shape[0]  # M
        self.num_latent_funcs = len(kern_list) # Q
        self.num_output_funcs = likelihood.num_output_functions(self.Y_metadata)
        if W_list is None:
            self.W_list, self.kappa_list = util.random_W_kappas(self.num_latent_funcs, self.num_output_funcs, rank=1)
        else:
            self.W_list = W_list
            _, self.kappa_list = util.random_W_kappas(self.num_latent_funcs, self.num_output_funcs, rank=1)

        self.Xmulti = X
        self.Ymulti = Y

        # Batch the data
        self.Xmulti_all, self.Ymulti_all = X, Y
        if batch_size is None:
            self.stochastic = False
            Xmulti_batch, Ymulti_batch = X, Y
        else:
            # Makes a climin slicer to make drawing minibatches much quicker
            self.stochastic = True
            self.slicer_list = []
            [self.slicer_list.append(draw_mini_slices(Xmulti_task.shape[0], self.batch_size)) for Xmulti_task in self.Xmulti]
            Xmulti_batch, Ymulti_batch = self.new_batch()
            self.Xmulti, self.Ymulti = Xmulti_batch, Ymulti_batch

        # Initialize inducing points Z
        #Z = kmm_init(self.X_all, self.num_inducing)
        self.Xdim = Z.shape[1]
        Z = np.tile(Z,(1,self.num_latent_funcs))

        inference_method = SVMOGPInf()

        super(SVMOGP, self).__init__(X=Xmulti_batch[0][1:10], Y=Ymulti_batch[0][1:10], Z=Z, kernel=kern_list[0], likelihood=likelihood,
                                     mean_function=None, X_variance=None, inference_method=inference_method,
                                     Y_metadata=Y_metadata, name=name, normalizer=False)

        self.unlink_parameter(self.kern)  # Unlink SparseGP default param kernel

        _, self.B_list = util.LCM(input_dim=self.Xdim, output_dim=self.num_output_funcs, rank=1, kernels_list=self.kern_list,
                                  W_list=self.W_list, kappa_list=self.kappa_list)

        # Set-up optimization parameters: [Z, m_u, L_u]
        self.q_u_means = Param('m_u', 2.5*np.random.randn(self.num_inducing, self.num_latent_funcs) +
                               0*np.tile(np.random.randn(1,self.num_latent_funcs),(self.num_inducing,1)))
        chols = choleskies.triang_to_flat(np.tile(np.eye(self.num_inducing)[None,:,:], (self.num_latent_funcs,1,1)))
        self.q_u_chols = Param('L_u', chols)

        self.link_parameter(self.Z, index=0)
        self.link_parameter(self.q_u_means)
        self.link_parameters(self.q_u_chols)
        [self.link_parameter(kern_q) for kern_q in kern_list]  # link all kernels
        [self.link_parameter(B_q) for B_q in self.B_list]

        self.vem_step = True # [True=VE-step, False=VM-step]
        self.ve_count = 0
        self.elbo = np.zeros((1,1))


    def log_likelihood(self):
        return self._log_marginal_likelihood

    def parameters_changed(self):
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()
        T = len(self.likelihood.likelihoods_list)
        self.batch_scale = []
        [self.batch_scale.append(float(self.Xmulti_all[t].shape[0] / self.Xmulti[t].shape[0])) for t in range(T)]
        self._log_marginal_likelihood, gradients, self.posteriors, _ = self.inference_method.inference(q_u_means=self.q_u_means,
                                                                        q_u_chols=self.q_u_chols, X=self.Xmulti, Y=self.Ymulti, Z=self.Z,
                                                                        kern_list=self.kern_list, likelihood=self.likelihood,
                                                                        B_list=self.B_list, Y_metadata=self.Y_metadata, batch_scale=self.batch_scale)
        D = self.likelihood.num_output_functions(self.Y_metadata)
        N = self.X.shape[0]
        M = self.num_inducing
        _, B_list = util.LCM(input_dim=self.Xdim, output_dim=D, rank=1, kernels_list=self.kern_list, W_list=self.W_list,
                             kappa_list=self.kappa_list)
        Z_grad = np.zeros_like(self.Z.values)
        for q, kern_q in enumerate(self.kern_list):
            # Update the variational parameter gradients:
            # SVI + VEM
            if self.stochastic:
                if self.vem_step:
                    self.q_u_means[:, q:q + 1].gradient = gradients['dL_dmu_u'][q]
                    self.q_u_chols[:, q:q + 1].gradient = gradients['dL_dL_u'][q]
                else:
                    self.q_u_means[:, q:q+1].gradient = np.zeros(gradients['dL_dmu_u'][q].shape)
                    self.q_u_chols[:,q:q+1].gradient = np.zeros(gradients['dL_dL_u'][q].shape)
            else:
                self.q_u_means[:, q:q + 1].gradient = gradients['dL_dmu_u'][q]
                self.q_u_chols[:, q:q + 1].gradient = gradients['dL_dL_u'][q]

            # Update kernel hyperparameters: lengthscale and variance
            kern_q.update_gradients_full(gradients['dL_dKmm'][q], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim])
            grad = kern_q.gradient.copy()

            # Update kernel hyperparameters: W + kappa
            Kffdiag = []
            KuqF = []
            for d in range(D):
                Kffdiag.append(gradients['dL_dKdiag'][q][d])
                KuqF.append(gradients['dL_dKmn'][q][d]*kern_q.K(self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]]))

            util.update_gradients_diag(self.B_list[q], Kffdiag)
            Bgrad = self.B_list[q].gradient.copy()
            util.update_gradients_Kmn(self.B_list[q], KuqF, D)
            Bgrad += self.B_list[q].gradient.copy()
            # SVI + VEM
            if self.stochastic:
                if self.vem_step:
                    self.B_list[q].gradient = np.zeros(Bgrad.shape)
                else:
                    self.B_list[q].gradient = Bgrad
            else:
                self.B_list[q].gradient = Bgrad

            for d in range(self.likelihood.num_output_functions(self.Y_metadata)):
                kern_q.update_gradients_full(gradients['dL_dKmn'][q][d], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]])
                grad += B_list[q].W[d]*kern_q.gradient.copy()
                kern_q.update_gradients_diag(gradients['dL_dKdiag'][q][d], self.Xmulti[f_index[d]])
                grad += B_list[q].B[d,d]*kern_q.gradient.copy()
                # SVI + VEM
                if self.stochastic:
                    if self.vem_step:
                        kern_q.gradient = np.zeros(grad.shape)
                    else:
                        kern_q.gradient = grad
                else:
                    kern_q.gradient = grad

            if not self.Z.is_fixed:
                Z_grad[:,q*self.Xdim:q*self.Xdim+self.Xdim] += kern_q.gradients_X(gradients['dL_dKmm'][q], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim])
                for d in range(self.likelihood.num_output_functions(self.Y_metadata)):
                    Z_grad[:,q*self.Xdim:q*self.Xdim+self.Xdim] += B_list[q].W[d]*kern_q.gradients_X(gradients['dL_dKmn'][q][d], self.Z[:,q*self.Xdim:q*self.Xdim+self.Xdim], self.Xmulti[f_index[d]])

        if not self.Z.is_fixed:
            # SVI + VEM
            if self.stochastic:
                if self.vem_step:
                    self.Z.gradient[:] = np.zeros(Z_grad.shape)
                else:
                    self.Z.gradient[:] = Z_grad
            else:
                self.Z.gradient[:] = Z_grad

    def set_data(self, X, Y):
        """
        Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients
        """
        self.Xmulti, self.Ymulti = X, Y

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

    def stochastic_grad(self, parameters):
        self.set_data(*self.new_batch())
        stochastic_gradients = self._grads(parameters)
        if self.vem_step:
            if self.ve_count > 2:
                self.ve_count = 0
                self.vem_step = False
            else:
                self.ve_count += 1
        else:
            self.vem_step = True
        return stochastic_gradients

    def callback(self, i, max_iter, verbose=True, verbose_plot=False):
        ll = self.log_likelihood()
        self.elbo[i['n_iter']-1,0] =  self.log_likelihood()[0]
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

    def _raw_predict(self, Xnew, latent_function_ind=None, full_cov=False, kern=None):
        """
        Make a prediction for the latent function values.

        For certain inputs we give back a full_cov of shape NxN,
        if there is missing data, each dimension has its own full_cov of shape NxNxD, and if full_cov is of,
        we take only the diagonal elements across N.

        For uncertain inputs, the SparseGP bound produces a full covariance structure across D, so for full_cov we
        return a NxDxD matrix and in the not full_cov case, we return the diagonal elements across D (NxD).
        This is for both with and without missing data. See for missing data SparseGP implementation py:class:'~GPy.models.sparse_gp_minibatch.SparseGPMiniBatch'.
        """
        #Plot f by default
        if latent_function_ind is None:
            latent_function_ind = 0

        if kern is None:
            kern = self.kern_list[latent_function_ind]

        posterior = self.posteriors[latent_function_ind]

        Kx = kern.K(self.Z, Xnew)
        mu = np.dot(Kx.T, posterior.woodbury_vector)
        if full_cov:
            Kxx = kern.K(Xnew)
            if posterior.woodbury_inv.ndim == 2:
                var = Kxx - np.dot(Kx.T, np.dot(posterior.woodbury_inv, Kx))
            elif posterior.woodbury_inv.ndim == 3:
                var = Kxx[:,:,None] - np.tensordot(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx).T, Kx, [1,0]).swapaxes(1,2)
            var = var
        else:
            Kxx = kern.Kdiag(Xnew)
            var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T

        return mu, np.abs(var) # corregir

    def _raw_predict_f(self, Xnew, output_function_ind=None, kern_list=None):
        f_ind = self.Y_metadata['function_index'].flatten()
        if output_function_ind is None:
            output_function_ind = 0
        d = output_function_ind
        if kern_list is None:
            kern_list = self.kern_list

        _,_,_,posteriors_F = self.inference_method.inference(q_u_means=self.q_u_means,
                                                       q_u_chols=self.q_u_chols, X=self.Xmulti_all, Y=self.Ymulti_all, Z=self.Z,
                                                       kern_list=self.kern_list, likelihood=self.likelihood,
                                                       B_list=self.B_list, Y_metadata=self.Y_metadata)
        posterior = posteriors_F[output_function_ind]
        Kx= np.zeros((self.Xmulti_all[f_ind[d]].shape[0], Xnew.shape[0]))
        Kxx = np.zeros((Xnew.shape[0], Xnew.shape[0]))
        for q, B_q in enumerate(self.B_list):
            Kx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(self.Xmulti_all[f_ind[d]], Xnew)
            Kxx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(Xnew, Xnew)

        mu = np.dot(Kx.T, posterior.woodbury_vector)
        Kxx = np.diag(Kxx)
        var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T

        return mu, np.abs(var) # corregir

    def predictive_new(self, Xnew, output_function_ind=None, kern_list=None):
        f_ind = self.Y_metadata['function_index'].flatten()
        if output_function_ind is None:
            output_function_ind = 0
        d = output_function_ind
        if kern_list is None:
            kern_list = self.kern_list

        Xmulti_all_new = self.Xmulti_all.copy()
        Xmulti_all_new[f_ind[d]] = Xnew

        posteriors_F = self.inference_method.inference(q_u_means=self.q_u_means,
                                                       q_u_chols=self.q_u_chols, X=Xmulti_all_new, Y=self.Ymulti_all, Z=self.Z,
                                                       kern_list=self.kern_list, likelihood=self.likelihood,
                                                       B_list=self.B_list, Y_metadata=self.Y_metadata, predictive=True)
        posterior = posteriors_F[output_function_ind]
        Kx= np.zeros((Xmulti_all_new[f_ind[d]].shape[0], Xnew.shape[0]))
        Kxx = np.zeros((Xnew.shape[0], Xnew.shape[0]))
        for q, B_q in enumerate(self.B_list):
            Kx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(Xmulti_all_new[f_ind[d]], Xnew)
            Kxx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(Xnew, Xnew)

        mu = np.dot(Kx.T, posterior.woodbury_vector)
        Kxx = np.diag(Kxx)
        var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T

        return mu, np.abs(var) # corregir

    def _raw_predict_stochastic(self, Xnew, output_function_ind=None, kern_list=None):
        f_ind = self.Y_metadata['function_index'].flatten()
        if output_function_ind is None:
            output_function_ind = 0
        d = output_function_ind
        if kern_list is None:
            kern_list = self.kern_list

        _,_,_,posteriors_F = self.inference_method.inference(q_u_means=self.q_u_means,
                                                       q_u_chols=self.q_u_chols, X=self.Xmulti_all, Y=self.Ymulti_all, Z=self.Z,
                                                       kern_list=self.kern_list, likelihood=self.likelihood,
                                                       B_list=self.B_list, Y_metadata=self.Y_metadata)
        posterior = posteriors_F[output_function_ind]
        Kx= np.zeros((self.Xmulti_all[f_ind[d]].shape[0], Xnew.shape[0]))
        Kxx = np.zeros((Xnew.shape[0], Xnew.shape[0]))
        for q, B_q in enumerate(self.B_list):
            Kx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(self.Xmulti_all[f_ind[d]], Xnew)
            Kxx += B_q.B[output_function_ind, output_function_ind] * kern_list[q].K(Xnew, Xnew)

        mu = np.dot(Kx.T, posterior.woodbury_vector)
        Kxx = np.diag(Kxx)
        var = (Kxx - np.sum(np.dot(np.atleast_3d(posterior.woodbury_inv).T, Kx) * Kx[None,:,:], 1)).T

        return mu, np.abs(var) # fix?

    def predictive(self, Xpred):
        D = self.num_output_funcs
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()
        m_F_pred = []
        v_F_pred = []
        for t in range(len(self.likelihood.likelihoods_list)):
            _,num_f_task,_ = self.likelihood.likelihoods_list[t].get_metadata()
            m_task_pred = np.empty((Xpred[t].shape[0], num_f_task))
            v_task_pred = np.empty((Xpred[t].shape[0], num_f_task))
            for d in range(D):
                if f_index[d] == t:
                    m_task_pred[:,d_index[d],None], v_task_pred[:,d_index[d],None] = self._raw_predict_f(Xpred[f_index[d]], output_function_ind=d)

            m_F_pred.append(m_task_pred)
            v_F_pred.append(v_task_pred)

        m_pred, v_pred = self.likelihood.predictive(m_F_pred, v_F_pred, self.Y_metadata)
        return m_pred, v_pred

    def negative_log_predictive(self, Xtest, Ytest, num_samples=1000):
        f_index = self.Y_metadata['function_index'].flatten()
        T = len(self.Ymulti)
        mu_F_star = []
        v_F_star = []
        for t in range(T):
            mu_F_star_task = np.empty((Ytest[t].shape[0],1))
            v_F_star_task = np.empty((Ytest[t].shape[0], 1))
            for d in range(self.num_output_funcs):
                if f_index[d] == t:
                    m_fd_star, v_fd_star = self._raw_predict_f(Xtest[t], output_function_ind=d)
                    mu_F_star_task = np.hstack((mu_F_star_task, m_fd_star))
                    v_F_star_task = np.hstack((v_F_star_task, v_fd_star))

            mu_F_star.append(mu_F_star_task[:,1:])
            v_F_star.append(v_F_star_task[:,1:])

        return self.likelihood.negative_log_predictive(Ytest, mu_F_star, v_F_star, Y_metadata=self.Y_metadata, num_samples=num_samples)

    def plot_u(self, dim=0, median=False, true_U=None, true_UX=None):
        """
        Plotting for models with two latent functions, one is an exponent over the scale
        parameter
        """
        Npred = 200 # number of predictive points
        if median:
            XX = fixed_inputs(self, non_fixed_inputs=[dim], fix_routine='median', as_list=False)
        else:
            XX = np.linspace(self.X_all[:, dim].min(), self.X_all[:, dim].max(), Npred)[:, None]
        X_pred_points = XX.copy()
        X_pred_points_lin = np.linspace(self.X_all[:, dim].min(), self.X_all[:, dim].max(), Npred)
        X_pred_points[:, dim] = X_pred_points_lin

        Q = self.num_latent_funcs
        m_q = np.empty((Npred, Q))
        v_q = np.empty((Npred, Q))

        for q in range(Q):
            m_q[:,q,None], v_q[:,q,None] = self._raw_predict(X_pred_points, latent_function_ind=q)

        u_q_std = np.sqrt(v_q)
        m_q_lower = m_q - 2*u_q_std
        m_q_upper = m_q + 2*u_q_std

        fig, ax = plt.subplots(figsize=(10, 6))
        X_dim = X_pred_points[:,dim:dim+1]
        for q in range(Q):
            plt.plot(X_dim, m_q[:,q], 'r-', alpha=0.25)
            plt.plot(X_dim, m_q_upper, 'b-', alpha=0.25)
            plt.plot(X_dim, m_q_lower, 'b-', alpha=0.25)

        if true_U is not None:
            plt.plot(true_UX, true_U, 'k+', alpha=0.5)
        plt.show()

    def plot_f(self, dim=0, median=False, true_F=None, true_FX=None):
        """
        Plotting for models with all output latent functions, one is an exponent over the scale
        parameter
        """
        Npred = 200 # number of predictive points
        f_index = self.Y_metadata['function_index'].flatten()
        d_index = self.Y_metadata['d_index'].flatten()

        D = self.num_output_funcs
        fig, ax = plt.subplots(figsize=(10, 6))
        for d in range(D):
            X_pred_points = np.linspace(self.Xmulti_all[f_index[d]][:, dim].min(), self.Xmulti_all[f_index[d]][:, dim].max(), Npred)[:,None]
            m_fd, v_fd = self._raw_predict_f(X_pred_points, output_function_ind=d)
            u_fd_std = np.sqrt(v_fd)
            m_fd_lower = m_fd - 2 * u_fd_std
            m_fd_upper = m_fd + 2 * u_fd_std

            plt.plot(X_pred_points, m_fd, 'r-', alpha=0.25)
            plt.plot(X_pred_points, m_fd_upper, 'b-', alpha=0.25)
            plt.plot(X_pred_points, m_fd_lower, 'b-', alpha=0.25)

            if true_F is not None:
                plt.plot(true_FX[f_index[d]], true_F[f_index[d]][:,d_index[d]], 'k-', alpha=0.5)

        plt.show()

    def plot_pred(self, Xpred, trueY=None, task=0):
        f_ind = self.Y_metadata['function_index'].flatten()
        y_ind = self.Y_metadata['y_index'].flatten()
        p_ind = self.Y_metadata['pred_index'].flatten()
        d_ind = self.Y_metadata['d_index'].flatten()
        m_pred, v_pred = self.predictive(Xpred)
        fig = plt.figure(figsize=(10, 6))
        if self.likelihood.ismulti(task):
            m_pred_mv = m_pred[task]
            Dt = m_pred_mv.shape[1]
            for d in range(Dt):
                plt.subplot(((Dt+1)*100) + 10 + d + 1)
                plt.plot(self.Xmulti_all[task], self.Ymulti_all[task], 'b+', alpha=0.75)
                if trueY is not None:
                    plt.plot(Xpred[task], trueY[task], 'b+', alpha=0.75)
                plt.plot(Xpred[task], m_pred[task][:,d], 'k-')

            plt.subplot(((Dt+1)*100) + 10 + Dt + 1)
            plt.plot(self.Xmulti_all[task], self.Ymulti_all[task], 'b+', alpha=0.75)
            plt.plot(Xpred[task], 1 - m_pred[task].sum(1), 'k-')

        else:
            std_pred = np.sqrt(v_pred[task])
            m_pred_lower = m_pred[task] - 2*std_pred
            m_pred_upper = m_pred[task] + 2*std_pred

            plt.plot(self.Xmulti_all[task], self.Ymulti_all[task], 'b+', alpha=0.75)
            if trueY is not None:
                plt.plot(Xpred[task], trueY[task], 'r+', alpha=0.75)

            plt.plot(Xpred[task], m_pred[task], 'k-')
            plt.plot(Xpred[task], m_pred_upper, 'k--', alpha=0.75)
            plt.plot(Xpred[task], m_pred_lower, 'k--', alpha=0.75)

        plt.show()
