import numpy as np
from KalmanFilter.core import KalmanFilter, KalmanMatrix
from tests.matrix_generation import generate_random_kalman_matrix

if __name__ == "__main__":

    # generate matrix for the process
    state_dim = 4
    observation_dim = 1
    noise_level = 0.0001

    state_transition_matrix = np.diag([0.1, 0.2, 0.4, 0.8])
    transition_noise_matrix = np.diag([0.8, 0.6, 0.5, 0.4])
    observation_output_matrix = np.array([1, 1, 1, 1]).reshape(observation_dim, state_dim)
    observation_noise_matrix = np.array([noise_level])
    initial_mean_matrix = np.array([0.25, 0.35, 0.3, 0.28])
    initial_covariance_matrix = np.diag([0.3, 0.3, 0.3, 0.3])

    kalman_matrix = KalmanMatrix(state_dim, observation_dim, state_transition_matrix, transition_noise_matrix,
                                 observation_output_matrix, observation_noise_matrix, initial_mean_matrix, initial_covariance_matrix)

    # generate the sequence using kalman matrix
    num_sample = 10000
    initial_state, state_sequence, observation_sequence = kalman_matrix.generate_sampled_sequence(num_sample)

    # generate kalman filter from kalman matrix
    kalman_filter = KalmanFilter(kalman_matrix)

    # generate the filtered state
    posterior_means, prior_means, posterior_covs, prior_covs = kalman_filter.forward_single_sequence(observation_sequence)

    # generate the filtered state
    smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = kalman_filter.backward_single_sequence(posterior_means, prior_means, posterior_covs, prior_covs)

    # calculate the correlation of prior means and posterior means
    for i in range(state_dim):
        corr_prior = np.corrcoef(np.ravel(prior_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        corr_posterior = np.corrcoef(np.ravel(posterior_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        corr_smooth = np.corrcoef(np.ravel(smooth_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        print("Correlation along state dim {} - prior mean: {:.2f}%, posterior mean: {:.2f}%, smooth mean: {:.2f}%".format(
            i, corr_prior * 100, corr_posterior * 100, corr_smooth * 100))


