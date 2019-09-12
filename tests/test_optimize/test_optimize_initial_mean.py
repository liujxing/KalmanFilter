import numpy as np
import os
import time
from KalmanFilter.kalman_matrix import KalmanMatrix
from KalmanFilter.kalman_optimizable_matrix import KalmanOptimizableMatrix
from KalmanFilter.kalman_optimizable_filter import KalmanOptimizableFilter
from tests.matrix_generation import generate_random_kalman_matrix, generate_optimizable_matrix_from_kalman_matrix

if __name__ == "__main__":

    # load the data if the data exist
    matrix_path = "data/test_optimize_initial_mean_matrix.npz"
    if os.path.exists(matrix_path):
        file = np.load(matrix_path)
        state_dim = file["state_transition_matrix"].shape[0],
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
    else:
        # generate matrix for the process
        state_dim = 4
        observation_dim = 2
        noise_level = 0.01
        kalman_matrix = generate_random_kalman_matrix(state_dim, observation_dim, noise_level)
        np.savez("data/test_optimize_initial_mean.npz",
                 state_transition_matrix=kalman_matrix.state_transition_matrix,
                 transition_noise_matrix=kalman_matrix.transition_noise_matrix,
                 observation_output_matrix=kalman_matrix.observation_output_matrix,
                 observation_noise_matrix=kalman_matrix.observation_noise_matrix,
                 initial_mean_matrix=kalman_matrix.initial_mean_matrix,
                 initial_covariance_matrix=kalman_matrix.initial_covariance_matrix,
                 )

    print("KalmanMatrix initial mean matrix:", np.ravel(kalman_matrix.initial_mean_matrix))
    kalman_optimizable_matrix = generate_optimizable_matrix_from_kalman_matrix(kalman_matrix, mask_initial_mean_matrix=True)
    print("KalmanOptimizableMatrix initial mean matrix: {}, optimize initial mean matrix: {}".format(
        kalman_optimizable_matrix.initial_mean_matrix, kalman_optimizable_matrix.optimize_initial_mean_matrix))

    # generate the sequence using kalman matrix
    data_path = "data/test_optimize_initial_mean_data.npz"
    if os.path.exists(data_path):
        file = np.load(data_path)
        initial_state_multiple_sequence = file["initial_state_multiple_sequence"]
        states_multiple_sequence = file["states_multiple_sequence"]
        observations_multiple_sequence = file["observations_multiple_sequence"]
        num_sequence = states_multiple_sequence.shape[0]
        sequence_length = states_multiple_sequence.shape[1]
    else:
        num_sequence = 10000
        sequence_length = 100

        initial_state_multiple_sequence = np.zeros((num_sequence, state_dim, 1))
        states_multiple_sequence = np.zeros((num_sequence, sequence_length, state_dim, 1))
        observations_multiple_sequence = np.zeros((num_sequence, sequence_length, observation_dim, 1))

        start_time = time.time()
        for i in range(num_sequence):
            if i % 100 == 0:
                print("Progress: {}/{}".format(i+1, num_sequence))
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

    # optimize the kalman matrix from data
    num_iteration = 20
    kalman_optimizable_filter = KalmanOptimizableFilter(kalman_optimizable_matrix)
    kalman_optimizable_filter.optimize_multiple_sequence(observations_multiple_sequence,
                                                         diagonal=False,
                                                         num_iteration=num_iteration)
    print("True value of initial_mean_matrix: {}".format(np.ravel(kalman_matrix.initial_mean_matrix)))
    print("Estimated value of initial_mean_matrix: {}".format(np.ravel(kalman_optimizable_matrix.initial_mean_matrix)))

