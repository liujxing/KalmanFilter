from KalmanFilter.kalman_filter import KalmanMatrix
import numpy as np


def generate_random_kalman_matrix(state_dim, observation_dim, noise_level=0.1) -> KalmanMatrix:

    state_transition_matrix = np.random.random((state_dim, state_dim)) - 0.5

    transition_noise_matrix = (np.random.random((state_dim, state_dim)) - 0.5) * noise_level
    transition_noise_matrix = transition_noise_matrix @ transition_noise_matrix.T

    observation_output_matrix = np.random.random((observation_dim, state_dim)) - 0.5

    observation_noise_matrix = (np.random.random((observation_dim, observation_dim)) - 0.5) * noise_level
    observation_noise_matrix = observation_noise_matrix @ observation_noise_matrix.T

    initial_mean_matrix = np.random.random((state_dim, 1)) - 0.5

    initial_covariance_matrix = (np.random.random((state_dim, state_dim)) - 0.5) * noise_level
    initial_covariance_matrix = initial_covariance_matrix @ initial_covariance_matrix.T

    # generate Kalman Matrix
    kalman_matrix = KalmanMatrix(state_dim, observation_dim,
                                 state_transition_matrix, transition_noise_matrix,
                                 observation_output_matrix, observation_noise_matrix,
                                 initial_mean_matrix, initial_covariance_matrix)

    return kalman_matrix

