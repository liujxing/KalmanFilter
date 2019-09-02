import numpy as np
from KalmanFilter.core import KalmanMatrix
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # generate matrix for the process
    state_dim = 4
    observation_dim = 2
    noise_level = 0.1

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

    # generate the sequence using kalman matrix
    num_sample = 1000
    initial_state, state_sequence, observation_sequence = kalman_matrix.generate_sampled_sequence(num_sample)

    print("initial state shape: {}".format(initial_state.shape))
    print("state_sequence shape: {}".format(state_sequence.shape))
    print("observation_sequence shape: {}".format(observation_sequence.shape))

    fig, ax_list = plt.subplots(state_dim + observation_dim)
    for i in range(state_dim):
        ax = ax_list[i]
        ax.plot(np.ravel(state_sequence[:, i]))
        ax.set_ylabel("state dim {}".format(i))
    for i in range(observation_dim):
        ax = ax_list[state_dim + i]
        ax.plot(np.ravel(observation_sequence[:, i]))
        ax.set_ylabel("observation dim {}".format(i))
    plt.show()




