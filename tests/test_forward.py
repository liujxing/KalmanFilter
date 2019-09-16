import numpy as np
from KalmanFilter.kalman_filter import KalmanFilter
from tests.matrix_generation import generate_random_kalman_matrix

if __name__ == "__main__":

    # generate matrix for the process
    state_dim = 4
    observation_dim = 2
    noise_level = 0.001
    kalman_matrix = generate_random_kalman_matrix(state_dim, observation_dim, noise_level)
    print("kalman_matrix is observable: {}".format(kalman_matrix.is_observable()))

    # generate the sequence using kalman matrix
    num_sample = 10000
    initial_state, state_sequence, observation_sequence = kalman_matrix.generate_sampled_sequence(num_sample)

    # generate kalman filter from kalman matrix
    kalman_filter = KalmanFilter(kalman_matrix)

    # generate the filtered state
    posterior_means, prior_means, posterior_covs, prior_covs = kalman_filter.forward_single_sequence(observation_sequence)

    # calculate the correlation of prior means and posterior means
    for i in range(state_dim):
        corr_prior = np.corrcoef(np.ravel(prior_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        corr_posterior = np.corrcoef(np.ravel(posterior_means[:, i]), np.ravel(state_sequence[:, i]))[0, 1]
        print("Correlation along state dim {} - prior mean: {:.2f}%, posterior mean: {:.2f}%".format(
            i, corr_prior * 100, corr_posterior * 100))


