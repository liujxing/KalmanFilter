from KalmanFilter.kalman_filter import KalmanMatrix
from KalmanFilter.kalman_optimizable_matrix import KalmanOptimizableMatrix
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


def generate_optimizable_matrix_from_kalman_matrix(kalman_matrix:KalmanMatrix,
                                                   mask_state_transition_matrix=False,
                                                   mask_transition_noise_matrix=False,
                                                   mask_observation_output_matrix=False,
                                                   mask_observation_noise_matrix=False,
                                                   mask_initial_mean_matrix=False,
                                                   mask_initial_covariance_matrix=False) -> KalmanOptimizableMatrix:
    """
    Generate a KalmanOptimizableMatrix based on given KalmanMatrix and selected matrix to mask for optimization
    :param kalman_matrix:
    :param mask_state_transition_matrix:
    :param mask_transition_noise_matrix:
    :param mask_observation_output_matrix:
    :param mask_observation_noise_matrix:
    :param mask_initial_mean_matrix:
    :param mask_initial_covariance_matrix:
    :return:
    """
    kalman_optimizable_matrix = KalmanOptimizableMatrix(kalman_matrix.get_state_dim(),
                                                        kalman_matrix.get_observation_dim(),
                                                        None if mask_state_transition_matrix else kalman_matrix.state_transition_matrix,
                                                        None if mask_transition_noise_matrix else kalman_matrix.transition_noise_matrix,
                                                        None if mask_observation_output_matrix else kalman_matrix.observation_output_matrix,
                                                        None if mask_observation_noise_matrix else kalman_matrix.observation_noise_matrix,
                                                        None if mask_initial_mean_matrix else kalman_matrix.initial_mean_matrix,
                                                        None if mask_initial_covariance_matrix else kalman_matrix.initial_covariance_matrix
                                                        )
    return kalman_optimizable_matrix


def generate_random_kalman_optimizable_matrix(state_dim, observation_dim, noise_level=0.1) -> KalmanOptimizableMatrix:
    """
    Generate a random KalmanOptimizableMatrix
    :param state_dim:
    :param observation_dim:
    :param noise_level:
    :return:
    """

    kalman_matrix = generate_random_kalman_matrix(state_dim, observation_dim, noise_level)
    return generate_optimizable_matrix_from_kalman_matrix(kalman_matrix)

