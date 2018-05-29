# Copyright (c) 2018 Pablo Moreno-Munoz
# Universidad Carlos III de Madrid and University of Sheffield

import numpy as np
from GPy.likelihoods import link_functions
from GPy.likelihoods import Likelihood
from itertools import compress


class HetLikelihood(Likelihood):
    """
    Heterogeneous Likelihood where

    """

    def __init__(self, likelihoods_list, gp_link=None ,name='heterogeneous_likelihood'):
        if gp_link is None:
            gp_link = link_functions.Identity()

        super(HetLikelihood, self).__init__(gp_link=gp_link, name=name)

        self.likelihoods_list = likelihoods_list

    def generate_metadata(self):
        """
        Generates Metadata: Given an Heterogeneous likelihood, it calculates the number functions f required in the
        model, the assignments of each f to its likelihood function, dimensionality of y_d and functions needed for
        predictions.
        """
        t_index = np.arange(len(self.likelihoods_list))
        y_index = np.empty((1,1))
        f_index = np.empty((1,1))
        d_index = np.empty((1,1))
        p_index = np.empty((1,1))
        for t, lik in enumerate(self.likelihoods_list):
            dim_y, dim_f, dim_pred = lik.get_metadata()
            y_index = np.hstack(( y_index, t*np.ones((1,dim_y)) ))
            f_index = np.hstack(( f_index, t*np.ones((1,dim_f)) ))
            d_index = np.hstack(( d_index, np.arange(0,dim_f)[None,:] ))
            p_index = np.hstack((p_index, t * np.ones((1, dim_pred))))

        metadata = {'task_index': t_index, 'y_index': np.int_(y_index[0,1:]), 'function_index': np.int_(f_index[0,1:]),
                    'd_index': np.int_(d_index[0,1:]),'pred_index': np.int_(p_index[0,1:])}
        return metadata

    def pdf(self, f, Y, Y_metadata):
        """
        Returns a list of PDFs from all likelihoods.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        tasks = np.unique(t_ind)
        pdf = np.zeros((Y.shape[0], t_ind.shape[0]))
        for t in tasks:
            pdf[:, t_ind == t] = self.likelihoods_list[t].pdf(f[:, f_ind == t], Y[:, y_ind == t], Y_metadata=None)
        return pdf

    def logpdf(self, f, Y, Y_metadata):
        """
        Returns a list of log-PDFs from all likelihoods.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        tasks = np.unique(t_ind)
        logpdf = np.zeros((Y.shape[0], t_ind.shape[0]))
        for t in tasks:
            logpdf[:, t_ind == t] = self.likelihoods_list[t].logpdf(f[:, f_ind == t], Y[:, y_ind == t], Y_metadata=None)
        return logpdf

    def samples(self, F, Y_metadata):
        """
        Returns a list of samples from all likelihoods.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        tasks = np.unique(t_ind)
        samples = []
        for t in tasks:
            samples.append(self.likelihoods_list[t].samples(F[t], num_samples=1, Y_metadata=None))
        return samples

    def num_output_functions(self, Y_metadata):
        """
        Returns the number of functions f that are required in the model for a given heterogeneous likelihood.
        """
        f_ind = Y_metadata['function_index'].flatten()
        return f_ind.shape[0]

    def num_latent_functions(self):
        pass

    def ismulti(self, task):
        """
        For a given task d (heterogeneous output) returns if y_d is or is not multivariate.
        """
        return self.likelihoods_list[task].ismulti()

    def var_exp(self, Y, mu_F, v_F, Y_metadata):
        """
        Returns a list of variational expectations from all likelihoods wrt to parameter functions (PFs) f.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        d_ind = Y_metadata['d_index'].flatten()
        tasks = np.unique(t_ind)
        var_exp = []
        for t in tasks:

            ve_task = self.likelihoods_list[t].var_exp(Y[t], mu_F[t], v_F[t], Y_metadata=None)
            var_exp.append(ve_task)
        return var_exp

    def var_exp_derivatives(self, Y, mu_F, v_F, Y_metadata):
        """
        Returns a list of variational expectations from all likelihood derivatives wrt to parameter functions (PFs) f.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        tasks = np.unique(t_ind)
        var_exp_dm = []
        var_exp_dv = []
        for t in tasks:
            ve_task_dm, ve_task_dv = self.likelihoods_list[t].var_exp_derivatives(Y[t], mu_F[t], v_F[t], Y_metadata=None)
            var_exp_dm.append(ve_task_dm)
            var_exp_dv.append(ve_task_dv)
        return var_exp_dm, var_exp_dv

    def predictive(self, mu_F_pred, v_F_pred, Y_metadata):
        """
        Returns a list of predictive mean and variance from all likelihoods.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        p_ind = Y_metadata['pred_index'].flatten()
        tasks = np.unique(t_ind)
        m_pred = []
        v_pred = []
        for t in tasks:
            m_pred_task, v_pred_task = self.likelihoods_list[t].predictive(mu_F_pred[t], v_F_pred[t], Y_metadata=None)
            m_pred.append(m_pred_task)
            v_pred.append(v_pred_task)
        return m_pred, v_pred

    def negative_log_predictive(self, Ytest, mu_F_star, v_F_star, Y_metadata, num_samples):
        """
        Returns the negative log-predictive density (NLPD) of the model over test data Ytest.
        """
        t_ind = Y_metadata['task_index'].flatten()
        y_ind = Y_metadata['y_index'].flatten()
        f_ind = Y_metadata['function_index'].flatten()
        p_ind = Y_metadata['pred_index'].flatten()
        tasks = np.unique(t_ind)
        logpred = 0
        for t in tasks:
            logpred += self.likelihoods_list[t].log_predictive(Ytest[t], mu_F_star[t], v_F_star[t], num_samples)

        nlogpred = -logpred
        return nlogpred
