import numpy as np
from typing import *


class KalmanMatrix(object):

    def __init__(self, state_dim, observation_dim,
                 state_transition_matrix=None, transition_noise_matrix=None,
                 observation_output_matrix=None, observation_noise_matrix=None,
                 initial_mean_matrix=None, initial_covariance_matrix=None):

        # perform dimension check
        if state_transition_matrix is not None:
            if state_transition_matrix.ndim != 2:
                raise ValueError("The dimension of state_transition_matrix should be 2 but get {}".format(state_transition_matrix.ndim))
            if state_transition_matrix.shape[0] != state_dim or state_transition_matrix.shape[1] != state_dim:
                raise ValueError("The shape of state_transition_matrix is {} but the state_dim is {}".format(state_transition_matrix.shape, state_dim))

        if transition_noise_matrix is not None:
            if transition_noise_matrix.ndim != 2:
                raise ValueError("The dimension of transition_noise_matrix should be 2 but get {}".format(transition_noise_matrix.ndim))
            if transition_noise_matrix.shape[0] != state_dim or transition_noise_matrix.shape[1] != state_dim:
                raise ValueError("The shape of transition_noise_matrix is {} but the state_dim is {}".format(transition_noise_matrix.shape, state_dim))

        if observation_output_matrix is not None:
            if observation_output_matrix.ndim == 1 and observation_dim == 1:
                observation_output_matrix = observation_output_matrix.reshape(1, -1)
            if observation_output_matrix.ndim != 2:
                raise ValueError("The dimension of observation_output_matrix should be 2 but get {}".format(observation_output_matrix.ndim))
            if observation_output_matrix.shape[0] != observation_dim or observation_output_matrix.shape[1] != state_dim:
                raise ValueError("The shape of observation_output_matrix is {} but expecting ({}, {})".format(observation_output_matrix.shape, observation_dim, state_dim))

        if observation_noise_matrix is not None:
            if observation_noise_matrix.ndim == 1:
                observation_noise_matrix = observation_noise_matrix.reshape(-1, 1)
            if observation_noise_matrix.ndim != 2:
                raise ValueError("The dimension of observation_noise_matrix should be 2 but get {}".format(observation_noise_matrix.ndim))
            if observation_noise_matrix.shape[0] != observation_dim or observation_noise_matrix.shape[1] != observation_dim:
                raise ValueError("The shape of observation_noise_matrix is {} but expecting ({}, {})".format(observation_noise_matrix.shape, observation_dim, observation_dim))

        if initial_mean_matrix is not None:
            if initial_mean_matrix.ndim == 1:
                initial_mean_matrix = initial_mean_matrix.reshape(-1, 1)
            if initial_mean_matrix.ndim != 2:
                raise ValueError("The dimension of initial_mean_matrix should be 2 but get {}".format(initial_mean_matrix.ndim))
            if initial_mean_matrix.shape[0] != state_dim or initial_mean_matrix.shape[1] != 1:
                raise ValueError("The shape of initial_mean_matrix is {} but expecting ({}, {})".format(initial_mean_matrix.shape, state_dim, 1))

        if initial_covariance_matrix is not None:
            if initial_covariance_matrix.ndim != 2:
                raise ValueError("The dimension of initial_covariance_matrix should be 2 but get {}".format(initial_covariance_matrix.ndim))
            if initial_covariance_matrix.shape[0] != state_dim or initial_covariance_matrix.shape[1] != state_dim:
                raise ValueError("The shape of initial_covariance_matrix is {} but expecting ({}, {})".format(initial_covariance_matrix.shape, state_dim, state_dim))

        # save the variables
        self.state_dim = state_dim
        self.observation_dim = observation_dim

        self.state_transition_matrix = state_transition_matrix
        self.transition_noise_matrix = transition_noise_matrix
        self.observation_output_matrix = observation_output_matrix
        self.observation_noise_matrix = observation_noise_matrix
        self.initial_mean_matrix = initial_mean_matrix
        self.initial_covariance_matrix = initial_covariance_matrix

    def get_state_dim(self):
        return self.state_dim

    def get_observation_dim(self):
        return self.observation_dim

    def _copy_optional_matrix(self, matrix:Optional[np.ndarray]):
        if matrix is None:
            return None
        else:
            return matrix.copy()

    def copy(self):
        matrix_copy = KalmanMatrix(self.state_dim, self.observation_dim,
                                   self._copy_optional_matrix(self.state_transition_matrix),
                                   self._copy_optional_matrix(self.transition_noise_matrix),
                                   self._copy_optional_matrix(self.observation_output_matrix),
                                   self._copy_optional_matrix(self.observation_noise_matrix),
                                   self._copy_optional_matrix(self.initial_mean_matrix),
                                   self._copy_optional_matrix(self.initial_covariance_matrix))
        return matrix_copy

    def has_none_matrix(self):
        return self.state_transition_matrix is None or self.transition_noise_matrix is None or \
            self.observation_output_matrix is None or self.observation_noise_matrix is None or \
            self.initial_mean_matrix is None or self.initial_covariance_matrix is None

    ####################### functions for sampling from sequence #########################
    def sample_state(self, prev_state):
        return self.state_transition_matrix @ prev_state + np.random.multivariate_normal(np.zeros(self.get_state_dim()), self.transition_noise_matrix).reshape(self.get_state_dim(), 1)

    def sample_observation(self, state):
        return self.observation_output_matrix @ state + np.random.multivariate_normal(np.zeros(self.get_observation_dim()), self.observation_noise_matrix).reshape(self.get_observation_dim(), 1)

    def generate_sampled_sequence(self, num_sample) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.has_none_matrix():
            raise ValueError("Cannot generate sequence when at least one of the matrix is None")
        initial_state = np.random.multivariate_normal(mean=np.ravel(self.initial_mean_matrix), cov=self.initial_covariance_matrix).reshape(self.get_state_dim(), 1)
        current_state = initial_state

        states = np.zeros((num_sample, self.get_state_dim(), 1))
        observations = np.zeros((num_sample, self.get_observation_dim(), 1))
        for i in range(num_sample):
            state = self.sample_state(current_state)
            observation = self.sample_observation(state)
            states[i] = state
            observations[i] = observation

            current_state = state

        return initial_state, states, observations

    ####################### functions for forward filtering #########################

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

    ####################### functions for backward smoothing #######################

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
        return self.get_smooth_mean(self.get_initial_forward_mean(), next_smooth_mean, next_prior_mean, backward_intermediate_matrix)
