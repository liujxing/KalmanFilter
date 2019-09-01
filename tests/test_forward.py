import numpy as np
from KalmanFilter.core import KalmanMatrix

if __name__ == "__main__":

    # generate matrix for the process
    num_hidden_dim = 4
    num_observation_dim = 2

    state_transition_matrix = (np.random.random((num_hidden_dim, num_hidden_dim)) - 0.5) * 1
    transition_noise_matrix = (np.random.random((num_hidden_dim, num_hidden_dim)) - 0.5) * 0.01
    transition_noise_matrix = transition_noise_matrix @ transition_noise_matrix.T

    observation_output_matrix = (np.random.random((num_observation_dim, num_hidden_dim)) - 0.5) * 1
    observation_noise_matrix = (np.random.random((num_observation_dim, num_observation_dim)) - 0.5) * 0.01
    observation_noise_matrix = observation_noise_matrix @ observation_noise_matrix.T

    initial_mean_matrix = (np.random.random((num_hidden_dim, 1)) - 0.5) * 1
    initial_covariance_matrix = (np.random.random((num_hidden_dim, num_hidden_dim)) - 0.5) * 0.01
    initial_covariance_matrix = initial_covariance_matrix @ initial_covariance_matrix.T

    # generate Kalman Matrix
    kalman_matrix = KalmanMatrix(state_transition_matrix, transition_noise_matrix,
                                 observation_output_matrix, observation_noise_matrix,
                                 initial_mean_matrix, initial_covariance_matrix
                                 )

    # generate random sequence from the kalman matrix
    num_sample = 10000

    initial_mean = kalman_matrix.get_initial_forward_mean()
    initial_cov = kalman_matrix.get_initial_forward_cov()

    x0 = np.random.multivariate_normal(np.ravel(initial_mean), initial_cov)
    x_samples = np.zeros((num_sample, num_hidden_dim))
    for i in range(num_sample):
        if i == 0:
            x_prev = x0
        else:
            x_prev = x_samples[i - 1]
        x_samples[i] = np.ravel(state_transition_matrix @ x_prev.reshape(-1, 1)) + np.random.multivariate_normal(
            np.zeros(num_hidden_dim), transition_noise_matrix)

    y_samples = np.zeros((num_sample, num_observation_dim))
    for i in range(num_sample):
        y_samples[i] = np.ravel(
            observation_output_matrix @ x_samples[i].reshape(-1, 1)) + np.random.multivariate_normal(
            np.zeros(num_observation_dim), observation_noise_matrix)

    # use Kalman filter to calculate the latent variable of the data
    x_results = np.zeros((num_sample, num_hidden_dim))

    prev_posterior_mean = kalman_matrix.get_initial_forward_mean()
    prev_posterior_cov = kalman_matrix.get_initial_forward_cov()

    for i in range(num_sample):
        current_prior_cov = kalman_matrix.get_next_prior_cov(prev_posterior_cov)
        current_gain_matrix = kalman_matrix.get_gain_matrix(current_prior_cov)
        current_posterior_cov = kalman_matrix.get_posterior_cov(current_prior_cov, current_gain_matrix)
        current_prior_mean = kalman_matrix.get_prior_mean(prev_posterior_mean)
        current_posterior_mean = kalman_matrix.get_posterior_mean(current_prior_mean, current_gain_matrix,
                                                                  y_samples[i].reshape(-1, 1))

        # set the result
        x_results[i] = np.ravel(current_posterior_mean)
        prev_posterior_mean = current_posterior_mean
        prev_posterior_cov = current_posterior_cov

    # calculate the correlation between states and mean obtained using Kalman filter

    for i in range(num_hidden_dim):
        print("Correlation along %dth hidden variable: %.2f%%" % (
            i, 100 * np.corrcoef(x_samples[:, i], x_results[:, i])[0, 1]))