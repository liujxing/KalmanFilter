import numpy as np
from KalmanFilter.kalman_matrix import KalmanMatrix
from KalmanFilter.kalman_filter import KalmanFilter


def load_kalman_matrix(matrix_path):

    # load the matrix file
    file = np.load(matrix_path)
    state_dim = file["state_transition_matrix"].shape[0]
    observation_dim = file["observation_output_matrix"].shape[0]
    kalman_matrix = KalmanMatrix(state_dim, observation_dim,
                                 file["state_transition_matrix"],
                                 file["transition_noise_matrix"],
                                 file["observation_output_matrix"],
                                 file["observation_noise_matrix"],
                                 file["initial_mean_matrix"],
                                 file["initial_covariance_matrix"]
                                 )
    file.close()
    return kalman_matrix


def load_data(data_path):

    # load the sample data file
    file = np.load(data_path)
    initial_state_multiple_sequence = file["initial_state_multiple_sequence"]
    states_multiple_sequence = file["states_multiple_sequence"]
    observations_multiple_sequence = file["observations_multiple_sequence"]
    file.close()
    return initial_state_multiple_sequence, states_multiple_sequence, observations_multiple_sequence


def smooth_multiple_sequence(kalman_matrix:KalmanMatrix, observations_multiple_sequence):

    # generate estimate of states using true matrix
    kalman_filter = KalmanFilter(kalman_matrix)
    state_dim = kalman_matrix.get_state_dim()
    num_sequence = observations_multiple_sequence.shape[0]
    sequence_length = observations_multiple_sequence.shape[1]

    smooth_initial_means_multiple_sequence = np.zeros((num_sequence, state_dim, 1))
    smooth_initial_covs_multiple_sequence = np.zeros((num_sequence, state_dim, state_dim))
    smooth_means_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, 1))
    smooth_covs_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, state_dim))
    smooth_lagged_covs_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, state_dim))
    for i in range(num_sequence):
        smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = kalman_filter.smooth_single_sequence(
            observations_multiple_sequence[i])
        smooth_initial_means_multiple_sequence[i] = smooth_mean_initial
        smooth_initial_covs_multiple_sequence[i] = smooth_cov_initial
        smooth_means_multiple_sequence[i] = smooth_means
        smooth_covs_multiple_sequence[i] = smooth_covs
        smooth_lagged_covs_multiple_sequence[i] = smooth_lagged_covs

    return smooth_initial_means_multiple_sequence, smooth_initial_covs_multiple_sequence, smooth_means_multiple_sequence, smooth_covs_multiple_sequence, smooth_lagged_covs_multiple_sequence