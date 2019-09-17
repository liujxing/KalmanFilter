import numpy as np
import time
from tests.matrix_generation import generate_random_kalman_matrix

if __name__ == "__main__":
    matrix_path = "data/kalman_matrix.npz"
    state_dim = 4
    observation_dim = 2
    noise_level = 0.1
    kalman_matrix = generate_random_kalman_matrix(state_dim, observation_dim, noise_level)
    np.savez(matrix_path,
             state_transition_matrix=kalman_matrix.state_transition_matrix,
             transition_noise_matrix=kalman_matrix.transition_noise_matrix,
             observation_output_matrix=kalman_matrix.observation_output_matrix,
             observation_noise_matrix=kalman_matrix.observation_noise_matrix,
             initial_mean_matrix=kalman_matrix.initial_mean_matrix,
             initial_covariance_matrix=kalman_matrix.initial_covariance_matrix,
             )

    # generate the sequence using kalman matrix
    data_path = "data/sample_data.npz"

    num_sequence = 10000
    sequence_length = 100

    initial_state_multiple_sequence = np.zeros((num_sequence, state_dim, 1))
    states_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, 1))
    observations_multiple_sequence = np.zeros((num_sequence, sequence_length, observation_dim, 1))

    start_time = time.time()
    for i in range(num_sequence):
        if i % 100 == 0:
            print("Progress: {}/{}".format(i + 1, num_sequence))
        initial_state, states, observations = kalman_matrix.generate_sampled_sequence(sequence_length)
        initial_state_multiple_sequence[i] = initial_state
        states_multiple_sequence[i] = states
        observations_multiple_sequence[i] = observations
    end_time = time.time()
    print("Time taken to generate the random sequence: %.2f seconds" % (end_time - start_time))
    np.savez(data_path,
             initial_state_multiple_sequence=initial_state_multiple_sequence,
             states_multiple_sequence=states_multiple_sequence,
             observations_multiple_sequence=observations_multiple_sequence)