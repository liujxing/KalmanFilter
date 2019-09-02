import numpy as np
from tests.matrix_generation import generate_random_kalman_matrix
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # generate matrix for the process
    state_dim = 4
    observation_dim = 2
    noise_level = 0.1
    kalman_matrix = generate_random_kalman_matrix(state_dim, observation_dim, noise_level)

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




