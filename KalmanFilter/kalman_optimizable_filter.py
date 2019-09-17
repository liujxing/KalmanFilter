import numpy as np
from typing import *
from KalmanFilter.kalman_filter import KalmanFilter
from KalmanFilter.kalman_optimizable_matrix import KalmanOptimizableMatrix

class KalmanOptimizableFilter(KalmanFilter):

    def __init__(self, kalman_matrix:KalmanOptimizableMatrix):
        super().__init__(kalman_matrix)

    ######################## functions for aggregated statistics #######################

    def get_G_matrix_single_sequence(self, observations):
        observation_dim = self.kalman_matrix.get_observation_dim()
        G_matrix = np.zeros((observation_dim, observation_dim))
        for i, observation in enumerate(observations):
            G_matrix += observation @ observation.T
        return G_matrix

    def get_G_matrix_multiple_sequences(self, observations_list):
        observation_dim = self.kalman_matrix.get_observation_dim()
        G_matrix = np.zeros((observation_dim, observation_dim))
        for observations in observations_list:
            G_matrix += self.get_G_matrix_single_sequence(observations)
        return G_matrix

    def get_D_matrix_single_sequence(self, observations, smooth_means):
        D_matrix = np.zeros((self.kalman_matrix.get_observation_dim(), self.kalman_matrix.get_state_dim()))
        for i, observation in enumerate(observations):
            smooth_mean = smooth_means[i]
            D_matrix += observation @ smooth_mean.T
        return D_matrix

    def get_D_matrix_multiple_sequences(self, observations_list, smooth_means_list):
        D_matrix = np.zeros((self.kalman_matrix.get_observation_dim(), self.kalman_matrix.get_state_dim()))
        for i in range(len(observations_list)):
            D_matrix ++ self.get_D_matrix_single_sequence(observations_list[i], smooth_means_list[i])
        return D_matrix

    def get_E_matrix_single_sequence(self, smooth_means, smooth_covs):
        state_dim = self.kalman_matrix.get_state_dim()
        E_matrix = np.zeros((state_dim, state_dim))
        for i in range(len(smooth_means)):
            smooth_mean = smooth_means[i]
            smooth_cov = smooth_covs[i]
            smooth_second_moment = self.kalman_matrix.get_smooth_second_moment(smooth_mean, smooth_cov)
            E_matrix += smooth_second_moment
        return E_matrix

    def get_E_matrix_multiple_sequences(self, smooth_means_list, smooth_covs_list):
        state_dim = self.kalman_matrix.get_state_dim()
        E_matrix = np.zeros((state_dim, state_dim))
        for i in range(len(smooth_means_list)):
            E_matrix += self.get_E_matrix_single_sequence(smooth_means_list[i], smooth_covs_list[i])
        return E_matrix

    def get_B_matrix_single_sequence(self, smooth_lagged_covs, smooth_means, smooth_mean_initial):
        state_dim = self.kalman_matrix.get_state_dim()
        B_matrix = np.zeros((state_dim, state_dim))
        for i in range(1, len(smooth_means)):
            smooth_mean = smooth_means[i]
            prev_smooth_mean = smooth_means[i - 1]
            smooth_lagged_cov = smooth_lagged_covs[i]
            B_matrix += smooth_lagged_cov + smooth_mean @ prev_smooth_mean.T
        B_matrix += smooth_lagged_covs[0] + smooth_means[0] @ smooth_mean_initial.T
        return B_matrix

    def get_B_matrix_multiple_sequences(self, smooth_lagged_covs_list, smooth_means_list, smooth_mean_initial_list):
        state_dim = self.kalman_matrix.get_state_dim()
        B_matrix = np.zeros((state_dim, state_dim))
        for i in range(len(smooth_lagged_covs_list)):
            B_matrix += self.get_B_matrix_single_sequence(smooth_lagged_covs_list[i], smooth_means_list[i], smooth_mean_initial_list[i])
        return B_matrix

    def get_A_matrix_single_sequence(self, E_matrix, smooth_means, smooth_covs, smooth_mean_initial, smooth_cov_initial):
        first_smooth_second_moment = self.kalman_matrix.get_smooth_second_moment(smooth_mean_initial,
                                                                                 smooth_cov_initial)
        last_smooth_second_moment = self.kalman_matrix.get_smooth_second_moment(smooth_means[-1], smooth_covs[-1])
        return E_matrix - last_smooth_second_moment + first_smooth_second_moment

    def get_A_matrix_multiple_sequences(self, E_matrix, smooth_means_list, smooth_covs_list, smooth_mean_initial_list, smooth_cov_initial_list):
        A_matrix = np.zeros(E_matrix.shape)
        for i in range(len(smooth_means_list)):
            A_matrix += self.get_A_matrix_single_sequence(E_matrix, smooth_means_list[i], smooth_covs_list[i], smooth_mean_initial_list[i], smooth_cov_initial_list[i])
        return A_matrix

    def get_num_sample(self, observations_list):
        num_sample = 0
        for observations in observations_list:
            num_sample += len(observations)
        return num_sample

    ################################## functions for getting kalman matrix ################################
    # TODO: find a better way of creating diagonal matrix
    def get_updated_state_transition_matrix(self, B_matrix, A_matrix, diagonal):
        if diagonal:
            result = np.zeros((self.kalman_matrix.get_state_dim(), self.kalman_matrix.get_state_dim()))
            np.fill_diagonal(result, np.diag(B_matrix) / np.diag(A_matrix))
        else:
            result = B_matrix @ np.linalg.inv(A_matrix)
        return result

    def get_updated_transition_noise_matrix(self, E_matrix, B_matrix, updated_state_transition_matrix,
                                            num_sample, diagonal):
        if diagonal:
            result = np.zeros((self.kalman_matrix.get_state_dim(), self.kalman_matrix.get_state_dim()))
            np.fill_diagonal(result, (np.diag(E_matrix) - np.diag(updated_state_transition_matrix) * np.diag(
                B_matrix)) / num_sample)
        else:
            result = (E_matrix - updated_state_transition_matrix @ B_matrix.T) / num_sample
        return result

    def get_updated_observation_output_matrix(self, D_matrix, E_matrix):
        return D_matrix @ np.linalg.inv(E_matrix)

    def get_updated_observation_noise_matrix(self, G_matrix, D_matrix, updated_observation_output_matrix,
                                             num_sample):
        result = (G_matrix - updated_observation_output_matrix @ D_matrix.T) / num_sample
        return result

    def get_updated_initial_mean_matrix_single_sequence(self, smooth_mean_initial):
        return smooth_mean_initial

    def get_updated_initial_mean_matrix_multiple_sequences(self, smooth_mean_initial_list):
        result = np.zeros((self.kalman_matrix.get_state_dim(), 1))
        for smooth_mean_initial in smooth_mean_initial_list:
            result += smooth_mean_initial
        result /= len(smooth_mean_initial_list)
        return result

    def get_updated_initial_cov_matrix_single_sequence(self, smooth_cov_initial, diagonal):
        if diagonal:
            diagonal = np.diag(smooth_cov_initial).copy()
            result = np.zeros(smooth_cov_initial.shape)
            np.fill_diagonal(result, diagonal)
        else:
            result = smooth_cov_initial
        return result

    # TODO: get the equation for the initial cov matrix of diagonal case
    def get_updated_initial_cov_matrix_multiple_sequences(self, smooth_cov_initial_list, smooth_mean_initial_list,
                                                          updated_initial_mean_matrix, diagonal):
        result = np.zeros((self.kalman_matrix.get_state_dim(), self.kalman_matrix.get_state_dim()))
        for i in range(len(smooth_cov_initial_list)):
            result += self.get_updated_initial_cov_matrix_single_sequence(smooth_cov_initial_list[i], diagonal)
            initial_mean_diff = smooth_mean_initial_list[i] - updated_initial_mean_matrix
            if diagonal:
                result += np.diag(initial_mean_diff**2)
            else:
                result += initial_mean_diff @ initial_mean_diff.T
        result /= len(smooth_cov_initial_list)
        return result




    ############################### function for optimizing using sequences ###########################

    def optimize_step_single_sequence(self, smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial,
                                      smooth_cov_initial, observations, diagonal):

        E_matrix = self.get_E_matrix_single_sequence(smooth_means, smooth_covs)
        D_matrix = self.get_D_matrix_single_sequence(observations, smooth_means)

        B_matrix = self.get_B_matrix_single_sequence(smooth_lagged_covs, smooth_means, smooth_mean_initial)
        A_matrix = self.get_A_matrix_single_sequence(E_matrix, smooth_means, smooth_covs, smooth_mean_initial, smooth_cov_initial)
        G_matrix = self.get_G_matrix_single_sequence(observations)

        num_sample = observations.shape[0]

        updated_state_transition_matrix = self.get_updated_state_transition_matrix(B_matrix, A_matrix, diagonal)
        updated_transition_noise_matrix = self.get_updated_transition_noise_matrix(E_matrix, B_matrix,
                                                                                   updated_state_transition_matrix,
                                                                                   num_sample, diagonal)
        updated_observation_output_matrix = self.get_updated_observation_output_matrix(D_matrix, E_matrix)
        updated_observation_noise_matrix = self.get_updated_observation_noise_matrix(G_matrix, D_matrix,
                                                                                     updated_observation_output_matrix,
                                                                                     num_sample)
        updated_initial_mean_matrix = self.get_updated_initial_mean_matrix_single_sequence(smooth_mean_initial)
        updated_initial_cov_matrix = self.get_updated_initial_cov_matrix_single_sequence(smooth_cov_initial, diagonal)

        return updated_state_transition_matrix, updated_transition_noise_matrix, updated_observation_output_matrix, \
               updated_observation_noise_matrix, updated_initial_mean_matrix, updated_initial_cov_matrix

    def optimize_step_multiple_sequences(self, smooth_means_list, smooth_covs_list, smooth_lagged_covs_list,
                                         smooth_mean_initial_list, smooth_cov_initial_list, observations_list, diagonal):

        E_matrix = self.get_E_matrix_multiple_sequences(smooth_means_list, smooth_covs_list)
        D_matrix = self.get_D_matrix_multiple_sequences(observations_list, smooth_means_list)

        B_matrix = self.get_B_matrix_multiple_sequences(smooth_lagged_covs_list, smooth_means_list, smooth_mean_initial_list)
        A_matrix = self.get_A_matrix_multiple_sequences(E_matrix, smooth_means_list, smooth_covs_list,
                                                        smooth_mean_initial_list, smooth_cov_initial_list)
        G_matrix = self.get_G_matrix_multiple_sequences(observations_list)

        num_sample = self.get_num_sample(observations_list)

        updated_state_transition_matrix = self.get_updated_state_transition_matrix(B_matrix, A_matrix, diagonal)
        updated_transition_noise_matrix = self.get_updated_transition_noise_matrix(E_matrix, B_matrix,
                                                                                   updated_state_transition_matrix,
                                                                                   num_sample, diagonal)
        updated_observation_output_matrix = self.get_updated_observation_output_matrix(D_matrix, E_matrix)
        updated_observation_noise_matrix = self.get_updated_observation_noise_matrix(G_matrix, D_matrix,
                                                                                     updated_observation_output_matrix,
                                                                                     num_sample)
        updated_initial_mean_matrix = self.get_updated_initial_mean_matrix_multiple_sequences(smooth_mean_initial_list)
        updated_initial_cov_matrix = self.get_updated_initial_cov_matrix_multiple_sequences(smooth_cov_initial_list, smooth_mean_initial_list, updated_initial_mean_matrix, diagonal)

        return updated_state_transition_matrix, updated_transition_noise_matrix, updated_observation_output_matrix, \
               updated_observation_noise_matrix, updated_initial_mean_matrix, updated_initial_cov_matrix

    def optimize_iteration_single_sequence(self, observations, diagonal):

        posterior_means, prior_means, posterior_covs, prior_covs = self.forward_single_sequence(observations)
        smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = self.backward_single_sequence(
            posterior_means, prior_means, posterior_covs, prior_covs)
        return self.optimize_step_single_sequence(smooth_means, smooth_covs, smooth_lagged_covs,
                                                  smooth_mean_initial, smooth_cov_initial, observations, diagonal)

    def optimize_iteration_multiple_sequences(self, observations_list, diagonal):
        # get the smooth parameters
        smooth_means_list = []
        smooth_covs_list = []
        smooth_lagged_covs_list = []
        smooth_mean_initial_list = []
        smooth_cov_initial_list = []
        for observations in observations_list:
            smooth_means, smooth_covs, smooth_lagged_covs, smooth_mean_initial, smooth_cov_initial = self.smooth_single_sequence(observations)
            smooth_means_list.append(smooth_means)
            smooth_covs_list.append(smooth_covs)
            smooth_lagged_covs_list.append(smooth_lagged_covs)
            smooth_mean_initial_list.append(smooth_mean_initial)
            smooth_cov_initial_list.append(smooth_cov_initial)
        return self.optimize_step_multiple_sequences(smooth_means_list, smooth_covs_list, smooth_lagged_covs_list,
                                                     smooth_mean_initial_list, smooth_cov_initial_list, observations_list, diagonal)



    @staticmethod
    def initialize_kalman_matrix_inplace(kalman_matrix: KalmanOptimizableMatrix, diagonal):
        if kalman_matrix.optimize_state_transition_matrix:
            kalman_matrix.state_transition_matrix = kalman_matrix.get_random_state_transition_matrix(
                diagonal=diagonal)

        if kalman_matrix.optimize_transition_noise_matrix:
            kalman_matrix.transition_noise_matrix = kalman_matrix.get_random_transition_noise_matrix(
                diagonal=diagonal)

        if kalman_matrix.optimize_observation_output_matrix:
            kalman_matrix.observation_output_matrix = kalman_matrix.get_random_observation_output_matrix()

        if kalman_matrix.optimize_observation_noise_matrix:
            kalman_matrix.observation_noise_matrix = kalman_matrix.get_random_observation_noise_matrix()

        if kalman_matrix.optimize_initial_mean_matrix:
            kalman_matrix.initial_mean_matrix = kalman_matrix.get_random_initial_mean_matrix()

        if kalman_matrix.optimize_initial_covariance_matrix:
            kalman_matrix.initial_covariance_matrix = kalman_matrix.get_random_initial_cov_matrix(
                diagonal=diagonal)

    @staticmethod
    def update_kalman_matrix_inplace(kalman_matrix, updated_state_transition_matrix, updated_transition_noise_matrix,
                                     updated_observation_output_matrix, updated_observation_noise_matrix,
                                     updated_initial_mean_matrix, updated_initial_cov_matrix):

        # update the parameters
        if kalman_matrix.optimize_state_transition_matrix:
            kalman_matrix.state_transition_matrix = updated_state_transition_matrix
        if kalman_matrix.optimize_transition_noise_matrix:
            kalman_matrix.transition_noise_matrix = updated_transition_noise_matrix
        if kalman_matrix.optimize_observation_output_matrix:
            kalman_matrix.observation_output_matrix = updated_observation_output_matrix
        if kalman_matrix.optimize_observation_noise_matrix:
            kalman_matrix.observation_noise_matrix = updated_observation_noise_matrix
        if kalman_matrix.optimize_initial_mean_matrix:
            kalman_matrix.initial_mean_matrix = updated_initial_mean_matrix
        if kalman_matrix.optimize_initial_covariance_matrix:
            kalman_matrix.initial_covariance_matrix = updated_initial_cov_matrix

    def optimize_single_sequence(self, observations, diagonal, num_iteration):

        self.initialize_kalman_matrix_inplace(self.kalman_matrix, diagonal)

        for i in range(num_iteration):
            updated_state_transition_matrix, updated_transition_noise_matrix, updated_observation_output_matrix, \
                updated_observation_noise_matrix, updated_initial_mean_matrix, updated_initial_cov_matrix = \
                self.optimize_iteration_single_sequence(observations, diagonal)

            self.update_kalman_matrix_inplace(self.kalman_matrix,
                                              updated_state_transition_matrix, updated_transition_noise_matrix,
                                              updated_observation_output_matrix, updated_observation_noise_matrix,
                                              updated_initial_mean_matrix, updated_initial_cov_matrix)

    def optimize_multiple_sequence(self, observations_list: List[np.ndarray], diagonal, num_iteration):

        self.initialize_kalman_matrix_inplace(self.kalman_matrix, diagonal)

        for i in range(num_iteration):
            updated_state_transition_matrix, updated_transition_noise_matrix, updated_observation_output_matrix, \
                updated_observation_noise_matrix, updated_initial_mean_matrix, updated_initial_cov_matrix = \
                self.optimize_iteration_multiple_sequences(observations_list, diagonal)

            self.update_kalman_matrix_inplace(self.kalman_matrix,
                                              updated_state_transition_matrix, updated_transition_noise_matrix,
                                              updated_observation_output_matrix, updated_observation_noise_matrix,
                                              updated_initial_mean_matrix, updated_initial_cov_matrix)




