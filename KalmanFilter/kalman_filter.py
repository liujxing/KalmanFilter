import numpy as np
from typing import *

from KalmanFilter.kalman_matrix import KalmanMatrix

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

        # check the shape of matrix
        if observations.ndim == 1 and self.kalman_matrix.get_observation_dim() == 1:
            observations = observations.expand_dims(1)
        if observations.ndim == 2:
            observations = observations.expand_dims(2)
        if observations.ndim != 3:
            raise ValueError("The dimension of observations should be 3, but get {}".format(observations.ndim))
        if observations.shape[1] != self.kalman_matrix.get_observation_dim() or observations.shape[2] != 1:
            raise ValueError("The shape of observations is {} but expecting (-1, {}, 1)".format(observations.shape, self.kalman_matrix.get_observation_dim()))

        state_dim = self.kalman_matrix.get_state_dim()
        num_sample = observations.shape[0]

        # generate array for storing the result
        posterior_means = np.zeros((num_sample, state_dim, 1))
        prior_means = np.zeros((num_sample, state_dim, 1))
        posterior_covs = np.zeros((num_sample, state_dim, state_dim))
        prior_covs = np.zeros((num_sample, state_dim, state_dim))

        current_posterior_mean = self.kalman_matrix.get_initial_forward_mean()
        current_posterior_cov = self.kalman_matrix.get_initial_forward_cov()

        for i in range(num_sample):
            posterior_mean, prior_mean, posterior_cov, prior_cov = self.forward_step(current_posterior_mean, current_posterior_cov, observations[i])

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
        num_sample = posterior_means.shape[0]

        # store the result
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

    def smooth_single_sequence(self, observations:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform smoothing of a single sequence
        :param observations:
        :return: smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial
        """

        posterior_means, prior_means, posterior_covs, prior_covs = self.forward_single_sequence(observations)
        smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = self.backward_single_sequence(
            posterior_means, prior_means, posterior_covs, prior_covs)
        return smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial





