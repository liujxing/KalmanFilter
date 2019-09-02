import numpy as np
from KalmanFilter.core import KalmanFilter, KalmanMatrix
from tests.matrix_generation import generate_random_kalman_matrix

if __name__ == "__main__":

    # generate matrix for the process
    state_dim = 4
    observation_dim = 2
    noise_level = 0.01
    kalman_matrix = generate_random_kalman_matrix(state_dim, observation_dim, noise_level)

    # generate the sequence using kalman matrix
    num_sample = 10000
    initial_state, state_sequence, observation_sequence = kalman_matrix.generate_sampled_sequence(num_sample)

    # generate kalman filter from kalman matrix
    kalman_filter = KalmanFilter(KalmanMatrix(state_dim, observation_dim))
    num_iteration = 10
    kalman_filter.optimize_single_sequence(observation_sequence, num_iteration)

    # generate the filtered state
    posterior_means, prior_means, posterior_covs, prior_covs = kalman_filter.forward_single_sequence(observation_sequence)

    # generate the smoothed state
    smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = kalman_filter.backward_single_sequence(posterior_means, prior_means, posterior_covs, prior_covs)

    # calculate the correlation of prior means and posterior means
    for i in range(state_dim):
        corr_prior = np.corrcoef(np.ravel(prior_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        corr_posterior = np.corrcoef(np.ravel(posterior_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        corr_smooth = np.corrcoef(np.ravel(smooth_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        print("Correlation along state dim {} - prior mean: {:.2f}%, posterior mean: {:.2f}%, smooth mean: {:.2f}%".format(
            i, corr_prior * 100, corr_posterior * 100, corr_smooth * 100))


