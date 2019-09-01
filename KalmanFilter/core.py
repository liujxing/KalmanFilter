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

    def get_backward_intermediate_matrix(self, posterior_cov, next_prior_cov):
        return posterior_cov @ self.state_transition_matrix.T @ np.linalg.inv(next_prior_cov)

    def get_smooth_mean(self, posterior_mean, next_smooth_mean, next_prior_mean, backward_intermediate_matrix):
        return posterior_mean + backward_intermediate_matrix @ (next_smooth_mean - next_prior_mean)

    def get_smooth_cov(self, posterior_cov, backward_intermediate_matrix, next_smooth_cov, next_prior_cov):
        return posterior_cov + backward_intermediate_matrix @ (next_smooth_cov - next_prior_cov) @ backward_intermediate_matrix.T

    def get_smooth_lagged_cov(self, next_smooth_cov, backward_intermediate_matrix):
        return next_smooth_cov @ backward_intermediate_matrix.T

    def get_initial_smooth_lagged_cov(self, next_smooth_cov, backward_intermediate_matrix):
        return self.get_smooth_lagged_cov(next_smooth_cov, backward_intermediate_matrix)

    def get_initial_smooth_cov(self, next_prior_cov, next_smooth_cov, backward_intermediate_matrix):
        return self.get_smooth_cov(self.get_initial_forward_cov(), backward_intermediate_matrix, next_smooth_cov, next_prior_cov)

    def get_initial_smooth_mean(self, next_smooth_mean, next_prior_mean, backward_intermediate_matrix):
        return self.get_smooth_mean(self.get_initial_forward_cov(), next_smooth_mean, next_prior_mean, backward_intermediate_matrix)

    # functions for optimization

    def get_smooth_second_moment(self, smooth_mean, smooth_cov):
        smooth_mean = smooth_mean.reshape(-1, 1)
        return smooth_cov + smooth_mean @ smooth_mean.T



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

    def backward_step_to_initial(self, next_prior_cov, next_smooth_mean, next_prior_mean, next_smooth_cov):
        backward_intermediate_matrix = self.kalman_matrix.get_backward_intermediate_matrix(self.kalman_matrix.get_initial_forward_cov(), next_prior_cov)
        smooth_mean_initial = self.kalman_matrix.get_initial_smooth_mean(next_smooth_mean, next_prior_mean, backward_intermediate_matrix)
        smooth_cov_initial = self.kalman_matrix.get_initial_smooth_cov(next_prior_cov, next_smooth_cov, backward_intermediate_matrix)
        smooth_lagged_cov_initial = self.kalman_matrix.get_initial_smooth_lagged_cov(next_smooth_cov, backward_intermediate_matrix)
        return smooth_mean_initial, smooth_cov_initial, smooth_lagged_cov_initial

    def backward_single_sequence(self, posterior_means, prior_means, posterior_covs, prior_covs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

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

        smooth_mean_initial, smooth_cov_initial, smooth_lagged_cov_initial = self.backward_step_to_initial(prior_covs[0], smooth_means[0], prior_means[0], smooth_covs[0])
        smooth_lagged_covs[0] = smooth_lagged_cov_initial

        return smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial


    def get_G_matrix(self, observations):
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        observation_dim = self.kalman_matrix.get_observation_dim()
        G_matrix = np.zeros((observation_dim, observation_dim))
        for i, observation in enumerate(observations):
            observation = observation.reshape(-1, 1)
            G_matrix += observation @ observation.T
        return G_matrix

    def get_D_matrix(self, observations, smooth_means):
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)

        observation_dim = self.kalman_matrix.get_observation_dim()
        D_matrix = np.zeros((observation_dim, observation_dim))
        for i, observation in enumerate(observations):
            observation = observation.reshape(-1, 1)
            smooth_mean = smooth_means[i].reshape(-1, 1)
            D_matrix += observation @ smooth_mean.T
        return D_matrix

    def get_E_matrix(self, smooth_means, smooth_covs):
        state_dim = self.kalman_matrix.get_state_dim()
        E_matrix = np.zeros((state_dim, state_dim))
        for i in range(len(smooth_means)):
            smooth_mean = smooth_means[i].reshape(-1, 1)
            smooth_cov = smooth_covs[i]
            smooth_second_moment = self.kalman_matrix.get_smooth_second_moment(smooth_mean, smooth_cov)
            E_matrix += smooth_second_moment
        return E_matrix

    def get_B_matrix(self, smooth_lagged_covs, smooth_means, smooth_mean_initial):
        state_dim = self.kalman_matrix.get_state_dim()
        B_matrix = np.zeros((state_dim, state_dim))
        for i in range(1, len(smooth_means)):
            smooth_mean = smooth_means[i].reshape(-1, 1)
            prev_smooth_mean = smooth_means[i-1].reshape(-1, 1)
            smooth_lagged_cov = smooth_lagged_covs[i]
            B_matrix += smooth_lagged_cov + smooth_mean @ prev_smooth_mean.T
        B_matrix += smooth_lagged_covs[0] + smooth_means[0].reshape(-1, 1) @ smooth_mean_initial.reshape(1, -1)
        return B_matrix

    def get_A_matrix(self, E_matrix, smooth_means, smooth_covs, smooth_mean_initial, smooth_cov_initial):
        first_smooth_second_moment = self.kalman_matrix.get_smooth_second_moment(smooth_mean_initial, smooth_cov_initial)
        last_smooth_second_moment = self.kalman_matrix.get_smooth_second_moment(smooth_means[-1].reshape(-1, 1), smooth_covs[-1])
        return E_matrix - last_smooth_second_moment + first_smooth_second_moment

    def get_updated_state_transition_matrix(self, B_matrix, A_matrix, num_sample):
        return B_matrix @ np.linalg.inv(A_matrix) / num_sample

    def get_updated_transition_noise_matrix(self, E_matrix, B_matrix, updated_state_transition_matrix, num_sample):
        return (E_matrix - updated_state_transition_matrix @ B_matrix.T) / num_sample

    def get_updated_observation_output_matrix(self, D_matrix, E_matrix, num_sample):
        return D_matrix @ np.linalg.inv(E_matrix) / num_sample

    def get_updated_observation_noise_matrix(self, G_matrix, D_matrix, updated_observation_output_matrix, num_sample):
        return (G_matrix - updated_observation_output_matrix @ D_matrix.T) / num_sample

    def get_updated_initial_mean_matrix(self, smooth_mean_initial):
        return smooth_mean_initial

    def get_updated_initial_cov_matrix(self, smooth_cov_initial):
        return smooth_cov_initial


    def optimize_step_single_sequence(self, smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial, observations):

        E_matrix = self.get_E_matrix(smooth_means, smooth_covs)
        D_matrix = self.get_D_matrix(observations, smooth_means)

        B_matrix = self.get_B_matrix(smooth_lagged_covs, smooth_means, smooth_mean_initial)
        A_matrix = self.get_A_matrix(E_matrix, smooth_means, smooth_covs, smooth_mean_initial, smooth_cov_initial)
        G_matrix = self.get_G_matrix(observations)

        num_sample = len(observations)

        updated_state_transition_matrix = self.get_updated_state_transition_matrix(B_matrix, A_matrix, num_sample)
        updated_transition_noise_matrix = self.get_updated_transition_noise_matrix(E_matrix, B_matrix, updated_state_transition_matrix, num_sample)
        updated_observation_output_matrix = self.get_updated_observation_output_matrix(D_matrix, E_matrix, num_sample)
        updated_observation_noise_matrix = self.get_updated_observation_noise_matrix(G_matrix, D_matrix, updated_observation_output_matrix, num_sample)
        updated_initial_mean_matrix = self.get_updated_initial_mean_matrix(smooth_mean_initial)
        updated_initial_cov_matrix = self.get_updated_initial_cov_matrix(smooth_cov_initial)

        self.kalman_matrix.state_transition_matrix = updated_state_transition_matrix
        self.kalman_matrix.transition_noise_matrix = updated_transition_noise_matrix
        self.kalman_matrix.observation_output_matrix = updated_observation_output_matrix
        self.kalman_matrix.observation_noise_matrix = updated_observation_noise_matrix
        self.kalman_matrix.initial_mean_matrix = updated_initial_mean_matrix
        self.kalman_matrix.initial_covariance_matrix = updated_initial_cov_matrix

        return self.kalman_matrix
