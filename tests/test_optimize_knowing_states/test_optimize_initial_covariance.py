import numpy as np
from KalmanFilter.kalman_matrix import KalmanMatrix
from KalmanFilter.kalman_filter import KalmanFilter
from KalmanFilter.kalman_optimizable_filter import KalmanOptimizableFilter
from tests.matrix_generation import generate_optimizable_matrix_from_kalman_matrix

if __name__ == "__main__":

    # load the matrix file
    matrix_path = "data/kalman_matrix.npz"
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

    # load the sample data file
    data_path = "data/sample_data.npz"
    file = np.load(data_path)
    initial_state_multiple_sequence = file["initial_state_multiple_sequence"]
    states_multiple_sequence = file["states_multiple_sequence"]
    observations_multiple_sequence = file["observations_multiple_sequence"]
    num_sequence = states_multiple_sequence.shape[0]
    sequence_length = states_multiple_sequence.shape[1]
    file.close()

    # generate estimate of states using true matrix
    smooth_states_path = "data/smooth_states.npz"
    file = np.load(smooth_states_path)
    smooth_initial_means_multiple_sequence = file["smooth_initial_means_multiple_sequence"]
    smooth_initial_covs_multiple_sequence = file["smooth_initial_covs_multiple_sequence"]
    smooth_means_multiple_sequence = file["smooth_means_multiple_sequence"]
    smooth_covs_multiple_sequence = file["smooth_covs_multiple_sequence"]
    smooth_lagged_covs_multiple_sequence = file["smooth_lagged_covs_multiple_sequence"]

    """
    kalman_filter = KalmanFilter(kalman_matrix)
    smooth_initial_means_multiple_sequence = np.zeros((num_sequence, state_dim, 1))
    smooth_initial_covs_multiple_sequence = np.zeros((num_sequence, state_dim, state_dim))
    smooth_means_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, 1))
    smooth_covs_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, state_dim))
    smooth_lagged_covs_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, state_dim))
    for i in range(num_sequence):
        smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = kalman_filter.smooth_single_sequence(observations_multiple_sequence[i])
        smooth_initial_means_multiple_sequence[i] = smooth_mean_initial
        smooth_initial_covs_multiple_sequence[i] = smooth_cov_initial
        smooth_means_multiple_sequence[i] = smooth_means
        smooth_covs_multiple_sequence[i] = smooth_covs
        smooth_lagged_covs_multiple_sequence[i] = smooth_lagged_covs

    np.savez(smooth_states_path,
             smooth_initial_means_multiple_sequence=smooth_initial_means_multiple_sequence,
             smooth_initial_covs_multiple_sequence=smooth_initial_covs_multiple_sequence,
             smooth_means_multiple_sequence=smooth_means_multiple_sequence,
             smooth_covs_multiple_sequence=smooth_covs_multiple_sequence,
             smooth_lagged_covs_multiple_sequence=smooth_lagged_covs_multiple_sequence)
    """

    # optimize the initial matrix using the states calculated from true matrix
    kalman_optimizable_matrix = generate_optimizable_matrix_from_kalman_matrix(kalman_matrix,
                                                                               mask_initial_covariance_matrix=True)
    kalman_optimizable_filter = KalmanOptimizableFilter(kalman_optimizable_matrix)
    new_initial_mean_matrix = kalman_optimizable_filter.get_updated_initial_mean_matrix_multiple_sequences(smooth_initial_means_multiple_sequence)
    print("true_initial_covariance_matrix - \n{}".format(kalman_matrix.initial_covariance_matrix))
    print("initial_covariance_matrix - before optimization: {}".format(kalman_optimizable_matrix.initial_covariance_matrix))
    kalman_optimizable_filter.optimize_multiple_sequence(observations_multiple_sequence, False, 1)
    print("new_initial_covariance_matrix - after optimization: \n{}".format(kalman_optimizable_filter.kalman_matrix.initial_covariance_matrix))

