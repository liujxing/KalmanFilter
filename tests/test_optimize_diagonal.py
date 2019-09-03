import numpy as np
from KalmanFilter.core import KalmanFilter, KalmanMatrix
from tests.matrix_generation import generate_random_kalman_matrix

if __name__ == "__main__":

    # generate matrix for the process
    state_dim = 4
    observation_dim = 1
    noise_level = 0.01

    state_transition_matrix = np.diag([0.1, 0.2, 0.4, 0.8])
    transition_noise_matrix = np.diag([0.8, 0.6, 0.5, 0.4])
    observation_output_matrix = np.array([1, 1, 1, 1]).reshape(observation_dim, state_dim)
    observation_noise_matrix = np.array([noise_level])
    initial_mean_matrix = np.array([0.25, 0.35, 0.3, 0.28])
    initial_covariance_matrix = np.diag([0.3, 0.3, 0.3, 0.3])

    kalman_matrix = KalmanMatrix(state_dim, observation_dim, state_transition_matrix, transition_noise_matrix,
                                 observation_output_matrix, observation_noise_matrix, initial_mean_matrix,
                                 initial_covariance_matrix)

    # generate the sequence using kalman matrix
    num_sample = 10000
    initial_state, state_sequence, observation_sequence = kalman_matrix.generate_sampled_sequence(num_sample)

    # generate kalman filter from kalman matrix
    kalman_filter = KalmanFilter(KalmanMatrix(state_dim, observation_dim,
                                              state_transition_matrix=state_transition_matrix,
                                              transition_noise_matrix=transition_noise_matrix,
                                              observation_output_matrix=observation_output_matrix,
                                              observation_noise_matrix=observation_noise_matrix,
                                              initial_mean_matrix=None,
                                              initial_covariance_matrix=initial_covariance_matrix,
                                              ))
    num_iteration = 10
    diagonal = True
    kalman_filter.optimize_single_sequence(observation_sequence, diagonal, num_iteration)

    # generate the filtered state
    #posterior_means, prior_means, posterior_covs, prior_covs = kalman_filter.forward_single_sequence(observation_sequence)

    # generate the smoothed state
    #smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = kalman_filter.backward_single_sequence(posterior_means, prior_means, posterior_covs, prior_covs)


    # compare the original matrix and the matrix from optimization
    print("True state transition matrix:\n", kalman_matrix.state_transition_matrix)
    print("Estimated state transition matrix:\n", kalman_filter.kalman_matrix.state_transition_matrix)

    print("True transition noise matrix:\n", kalman_matrix.transition_noise_matrix)
    print("Estimated transition noise matrix:\n", kalman_filter.kalman_matrix.transition_noise_matrix)

    print("True observation output matrix:\n", kalman_matrix.observation_output_matrix)
    print("Estimated observation output matrix:\n", kalman_filter.kalman_matrix.observation_output_matrix)

    print("True observation noise matrix:\n", kalman_matrix.observation_noise_matrix)
    print("Estimated observation noise matrix:\n", kalman_filter.kalman_matrix.observation_noise_matrix)

    print("True initial mean matrix:\n", kalman_matrix.initial_mean_matrix)
    print("Estimated initial mean matrix:\n", kalman_filter.kalman_matrix.initial_mean_matrix)
    print("True initial state:\n", state_sequence[0])

    print("True initial covariance matrix:\n", kalman_matrix.initial_covariance_matrix)
    print("Estimated initial covariance matrix:\n", kalman_filter.kalman_matrix.initial_covariance_matrix)




