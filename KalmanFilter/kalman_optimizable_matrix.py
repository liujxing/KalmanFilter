import numpy as np
from typing import *
from KalmanFilter.kalman_matrix import KalmanMatrix


class KalmanOptimizableMatrix(KalmanMatrix):

    def __init__(self, state_dim, observation_dim,
                 state_transition_matrix=None, transition_noise_matrix=None,
                 observation_output_matrix=None, observation_noise_matrix=None,
                 initial_mean_matrix=None, initial_covariance_matrix=None,
                 optimize_state_transition_matrix=None, optimize_transition_noise_matrix=None,
                 optimize_observation_output_matrix=None, optimize_observation_noise_matrix=None,
                 optimize_initial_mean_matrix=None, optimize_initial_covariance_matrix=None,
                 ):

        super().__init__(state_dim, observation_dim,
                         state_transition_matrix, transition_noise_matrix,
                         observation_output_matrix, observation_noise_matrix,
                         initial_mean_matrix, initial_covariance_matrix)

        # whether to optimize the matrix
        if optimize_state_transition_matrix is None:
            self.optimize_state_transition_matrix = state_transition_matrix is None
        else:
            self.optimize_state_transition_matrix = optimize_state_transition_matrix

        if optimize_transition_noise_matrix is None:
            self.optimize_transition_noise_matrix = transition_noise_matrix is None
        else:
            self.optimize_transition_noise_matrix = optimize_transition_noise_matrix

        if optimize_observation_output_matrix is None:
            self.optimize_observation_output_matrix = observation_output_matrix is None
        else:
            self.optimize_observation_output_matrix = optimize_observation_output_matrix

        if optimize_observation_noise_matrix is None:
            self.optimize_observation_noise_matrix = observation_noise_matrix is None
        else:
            self.optimize_observation_noise_matrix = optimize_observation_noise_matrix

        if optimize_initial_mean_matrix is None:
            self.optimize_initial_mean_matrix = initial_mean_matrix is None
        else:
            self.optimize_initial_mean_matrix = optimize_initial_mean_matrix

        if optimize_initial_covariance_matrix is None:
            self.optimize_initial_covariance_matrix = initial_covariance_matrix is None
        else:
            self.optimize_initial_covariance_matrix = optimize_initial_covariance_matrix

        ######################## functions for optimization #######################

    def get_smooth_second_moment(self, smooth_mean, smooth_cov):
        return smooth_cov + smooth_mean @ smooth_mean.T

    def fill_nondiagonal_elements(self, matrix, fill_value):
        diagonal = np.diag(matrix).copy()
        matrix.fill(fill_value)
        np.fill_diagonal(matrix, diagonal)
        return matrix

    def get_random_state_transition_matrix(self, diagonal=False):
        state_transition_matrix = np.random.random((self.get_state_dim(), self.get_state_dim())) - 0.5
        if diagonal:
            self.fill_nondiagonal_elements(state_transition_matrix, 0)
        return state_transition_matrix

    def get_random_transition_noise_matrix(self, noise_level=0.1, diagonal=False):
        transition_noise_matrix = (np.random.random((self.get_state_dim(), self.get_state_dim())) - 0.5) * noise_level
        transition_noise_matrix = transition_noise_matrix @ transition_noise_matrix.T
        if diagonal:
            self.fill_nondiagonal_elements(transition_noise_matrix, 0)
        return transition_noise_matrix

    def get_random_observation_output_matrix(self):
        observation_output_matrix = np.random.random((self.get_observation_dim(), self.get_state_dim())) - 0.5
        return observation_output_matrix

    def get_random_observation_noise_matrix(self, noise_level=0.1):
        observation_noise_matrix = (np.random.random(
            (self.get_observation_dim(), self.get_observation_dim())) - 0.5) * noise_level
        observation_noise_matrix = observation_noise_matrix @ observation_noise_matrix.T
        return observation_noise_matrix

    def get_random_initial_mean_matrix(self):
        return np.random.random((self.get_state_dim(), 1)) - 0.5

    def get_random_initial_cov_matrix(self, noise_level=0.1, diagonal=False):
        initial_covariance_matrix = (np.random.random((self.get_state_dim(), self.get_state_dim())) - 0.5) * noise_level
        initial_covariance_matrix = initial_covariance_matrix @ initial_covariance_matrix.T
        if diagonal:
            self.fill_nondiagonal_elements(initial_covariance_matrix, 0)
        return initial_covariance_matrix





