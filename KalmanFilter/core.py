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

    def get_state_dim(self):
        return len(self.transition_noise_matrix)

    def get_observation_dim(self):
        return len(self.observation_noise_matrix)

    # functions for forward filtering

    def get_initial_forward_mean(self) -> np.ndarray:
        return self.initial_mean_matrix

    def get_initial_forward_cov(self) -> np.ndarray:
        return self.initial_covariance_matrix

    def get_next_prior_cov(self, prev_posterior_cov):
        return self.state_transition_matrix @ prev_posterior_cov @ self.state_transition_matrix.T + self.transition_noise_matrix

    def get_forward_intermediate_matrix(self, prior_cov):
        return self.observation_output_matrix @ prior_cov @ self.observation_output_matrix.T + self.observation_noise_matrix

    def get_gain_matrix(self, prior_cov):
        intermediate_matrix = self.get_forward_intermediate_matrix(prior_cov)
        return prior_cov @ self.observation_output_matrix.T @ np.linalg.inv(intermediate_matrix)

    def get_posterior_cov(self, prior_cov, gain_matrix):
        return prior_cov - gain_matrix @ self.observation_output_matrix @ prior_cov

    def get_prior_mean(self, prev_posterior_mean):
        return self.state_transition_matrix @ prev_posterior_mean

    def get_posterior_mean(self, prior_mean, gain_matrix, observation):
        return prior_mean + gain_matrix @ (observation - self.observation_output_matrix @ prior_mean)

    # functions for backward smoothing

    def get_initial_backward_lagged_smooth_cov(self, gain_matrix, prev_posterior_cov):
        intermediate = self.state_transition_matrix @ prev_posterior_cov
        return intermediate - gain_matrix @ self.observation_output_matrix @ intermediate

    def get_backward_intermediate_matrix(self, posterior_cov, next_prior_cov):
        return posterior_cov @ self.state_transition_matrix.T @ np.linalg.inv(next_prior_cov)

    def get_smooth_mean(self, posterior_mean, next_smooth_mean, next_prior_mean, backward_intermediate_matrix):
        return posterior_mean + backward_intermediate_matrix @ (next_smooth_mean - next_prior_mean)

    def get_smooth_cov(self, posterior_cov, backward_intermediate_matrix, next_smooth_cov, next_prior_cov):
        return posterior_cov + backward_intermediate_matrix @ (next_smooth_cov - next_prior_cov) @ backward_intermediate_matrix.T

    def get_smooth_lagged_cov(self, next_smooth_cov, backward_intermediate_matrix):
        return next_smooth_cov @ backward_intermediate_matrix.T

    def get_initial_smooth_lagged_cov(self, next_prior_cov, next_smooth_cov):
        backward_intermediate_matrix = self.get_backward_intermediate_matrix(self.get_initial_forward_cov(), next_prior_cov)
        return self.get_smooth_lagged_cov(next_smooth_cov, backward_intermediate_matrix)




class KalmanFilter(object):

    def __init__(self, kalman_matrix:KalmanMatrix):
        self.kalman_matrix = kalman_matrix

    def forward_step(self, prev_posterior_mean, prev_posterior_cov, observation) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        prior_cov = self.kalman_matrix.get_next_prior_cov(prev_posterior_cov)
        gain_matrix = self.kalman_matrix.get_gain_matrix(prior_cov)
        posterior_cov = self.kalman_matrix.get_posterior_cov(prior_cov, gain_matrix)
        prior_mean = self.kalman_matrix.get_prior_mean(prev_posterior_mean)
        posterior_mean = self.kalman_matrix.get_posterior_mean(prior_mean, gain_matrix, observation)

        return posterior_mean, prior_mean, posterior_cov, prior_cov

    def forward_single_sequence(self, observations) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        state_dim = self.kalman_matrix.get_state_dim()
        num_sample = len(observations)

        posterior_means = np.zeros((num_sample, state_dim, 1))
        prior_means = np.zeros((num_sample, state_dim, 1))
        posterior_covs = np.zeros((num_sample, state_dim, state_dim))
        prior_covs = np.zeros((num_sample, state_dim, state_dim))

        current_posterior_mean = self.kalman_matrix.get_initial_forward_mean()
        current_posterior_cov = self.kalman_matrix.get_initial_forward_cov()

        for i in range(num_sample):
            posterior_mean, prior_mean, posterior_cov, prior_cov = self.forward_step(current_posterior_mean, current_posterior_cov, observations[i].reshape(-1, 1))

            prior_means[i] = prior_mean
            posterior_means[i] = posterior_mean
            posterior_covs[i] = posterior_cov
            prior_covs[i] = prior_cov

            current_posterior_mean = posterior_mean
            current_posterior_cov = posterior_cov

        return posterior_means, prior_means, posterior_covs, prior_covs

    def backward_step(self, posterior_mean, next_smooth_mean, next_prior_mean, posterior_cov, next_prior_cov, next_smooth_cov) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        backward_intermediate_matrix = self.kalman_matrix.get_backward_intermediate_matrix(posterior_cov, next_prior_cov)
        smooth_mean = self.kalman_matrix.get_smooth_mean(posterior_mean, next_smooth_mean, next_prior_mean, backward_intermediate_matrix)
        smooth_cov = self.kalman_matrix.get_smooth_cov(posterior_cov, backward_intermediate_matrix, next_smooth_cov, next_prior_cov)
        smooth_lagged_cov = self.kalman_matrix.get_smooth_lagged_cov(next_smooth_cov, backward_intermediate_matrix)
        return smooth_mean, smooth_cov, smooth_lagged_cov

    def backward_single_sequence(self, posterior_means, prior_means, posterior_covs, prior_covs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        state_dim = self.kalman_matrix.get_state_dim()
        num_sample = len(posterior_means)
        smooth_means = np.zeros((num_sample, state_dim, 1))
        smooth_covs = np.zeros((num_sample, state_dim, state_dim))
        smooth_lagged_covs = np.zeros((num_sample, state_dim, state_dim))

        current_smooth_mean = posterior_means[-1]
        current_smooth_cov = posterior_covs[-1]
        smooth_means[-1] = current_smooth_mean
        smooth_covs[-1] = current_smooth_cov

        for i in range(num_sample - 2, -1, -1):
            smooth_mean, smooth_cov, smooth_lagged_cov = self.backward_step(posterior_means[i], current_smooth_mean, prior_means[i+1],
                                                                            posterior_covs[i], prior_covs[i+1], current_smooth_cov)
            smooth_means[i] = smooth_mean
            smooth_covs[i] = smooth_cov
            smooth_lagged_covs[i+1] = smooth_lagged_cov

            current_smooth_mean = smooth_mean
            current_smooth_cov = smooth_cov

        smooth_lagged_covs[0] = self.kalman_matrix.get_initial_smooth_lagged_cov(prior_covs[0], smooth_covs[0])
        return smooth_means, smooth_covs, smooth_lagged_covs



    def em_step(self):
        pass


    def fit_single(self):
        pass


    def fit(self):
        pass

