import numpy as np
from typing import *


class KalmanMatrix(object):

    def __init__(self, state_transition_matrix=None, transition_noise_matrix=None,
                 observation_output_matrix=None, observation_noise_matrix=None,
                 initial_mean_matrix=None, initial_covariance_matrix=None):

        self.state_transition_matrix = state_transition_matrix
        self.transition_noise_matrix = transition_noise_matrix
        self.observation_output_matrix = observation_output_matrix
        self.observation_noise_matrix = observation_noise_matrix
        self.initial_mean_matrix = initial_mean_matrix
        self.initial_covariance_matrix = initial_covariance_matrix

    # functions for forward filtering

    def get_initial_forward_mean(self) -> np.ndarray:
        return self.initial_mean_matrix

    def get_initial_forward_cov(self) -> np.ndarray:
        return self.initial_covariance_matrix

    def get_next_prior_cov(self, prev_posterior_cov):
        return self.state_transition_matrix @ prev_posterior_cov @ self.state_transition_matrix.T + self.transition_noise_matrix

    def get_intermediate_matrix(self, prior_cov):
        return self.observation_output_matrix @ prior_cov @ self.observation_output_matrix.T + self.observation_noise_matrix

    def get_gain_matrix(self, prior_cov):
        intermediate_matrix = self.get_intermediate_matrix(prior_cov)
        return prior_cov @ self.observation_output_matrix.T @ np.linalg.inv(intermediate_matrix)

    def get_posterior_cov(self, prior_cov, gain_matrix):
        return prior_cov - gain_matrix @ self.observation_output_matrix @ prior_cov

    def get_posterior_mean(self, prev_posterior_mean, gain_matrix, observation):
        prior_mean = self.state_transition_matrix @ prev_posterior_mean
        return prior_mean + gain_matrix @ (observation - self.observation_output_matrix @ prior_mean)

    # functions for backward smoothing

    def get_initial_backward_mean(self, posterior_mean):
        return posterior_mean

    def get_initial_backward_cov(self, posterior_cov):
        return posterior_cov

    def get_initial_backward_lagged_cov(self, gain_matrix, prev_posterior_cov):
        intermediate = gain_matrix @ prev_posterior_cov
        return intermediate - gain_matrix @ self.observation_output_matrix @ intermediate



class KalmanFilter(object):


    def __init__(self):
        self.kalman_matrix = KalmanMatrix()


    def forward_step(self, prev_posterior_mean, prev_posterior_cov, observation) -> Tuple[np.ndarray, np.ndarray]:

        prior_cov = self.kalman_matrix.get_next_prior_cov(prev_posterior_cov)
        gain_matrix = self.kalman_matrix.get_gain_matrix(prior_cov)
        posterior_cov = self.kalman_matrix.get_posterior_cov(prior_cov, gain_matrix)
        posterior_mean = self.kalman_matrix.get_posterior_mean(prev_posterior_mean, gain_matrix, observation)

        return posterior_mean, posterior_cov

    def forward_step_initialization(self):

        return self.kalman_matrix.get_initial_forward_mean(), self.kalman_matrix.get_initial_forward_cov()




    def backward_step(self):
        pass


    def em_step(self):
        pass


    def fit_single(self):
        pass


    def fit(self):
        pass

