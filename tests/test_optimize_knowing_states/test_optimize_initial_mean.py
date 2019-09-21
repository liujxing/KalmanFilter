import numpy as np
from KalmanFilter.kalman_matrix import KalmanMatrix
from KalmanFilter.kalman_filter import KalmanFilter
from KalmanFilter.kalman_optimizable_filter import KalmanOptimizableFilter
from tests.matrix_generation import generate_optimizable_matrix_from_kalman_matrix
from tests.test_optimize_knowing_states.utility import *

if __name__ == "__main__":

    # load the matrix file
    matrix_path = "data/kalman_matrix.npz"
    kalman_matrix = load_kalman_matrix(matrix_path)

    # load the sample data file
    data_path = "data/sample_data.npz"
    initial_state_multiple_sequence, states_multiple_sequence, observations_multiple_sequence = load_data(data_path)

    # generate estimate of states using true matrix
    smooth_initial_means_multiple_sequence, smooth_initial_covs_multiple_sequence, smooth_means_multiple_sequence, smooth_covs_multiple_sequence, smooth_lagged_covs_multiple_sequence = smooth_multiple_sequence(kalman_matrix, observations_multiple_sequence)

    # optimize the initial matrix using the states calculated from true matrix
    kalman_optimizable_matrix = generate_optimizable_matrix_from_kalman_matrix(kalman_matrix,
                                                                               mask_initial_mean_matrix=True)
    kalman_optimizable_filter = KalmanOptimizableFilter(kalman_optimizable_matrix)
    new_initial_mean_matrix = kalman_optimizable_filter.get_mean_initial_means(smooth_initial_means_multiple_sequence)
    print("true_initial_mean_matrix - {}".format(np.ravel(kalman_matrix.initial_mean_matrix)))
    print("initial_mean_matrix - before optimization: {}".format(kalman_optimizable_matrix.initial_mean_matrix))
    print("new_initial_mean_matrix - optimization using function: {}".format(np.ravel(new_initial_mean_matrix)))
    total_iteration = 200
    for i in range(total_iteration):
        kalman_optimizable_filter.optimize_multiple_sequence(observations_multiple_sequence, False, 1)
        print("new_initial_mean_matrix - after optimization iteration {}/{}: {}".format(i, total_iteration, np.ravel(kalman_optimizable_matrix.initial_mean_matrix)))


