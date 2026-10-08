# mocaxextendpy - Python implementation of Tensor Extension Algorithms
# Copyright 2021 iRuiz Technologies, ltd.
# See LICENSE.txt for details on the End User License Agreement.

import numpy as np
import copy
import ctypes
from numpy.linalg import LinAlgError

from mocaxextendpy.mocax_extend_C_interface import hom_mat

from scikit_tt.tensor_train import TT


class AlternatingLeastSquares(object):
    """
        class that manages the routines associated to the Alternating Least Squares scheme used by the Completion Algorithm
    """

    @staticmethod
    def alternating_least_squares(tensor_entries_train,
                                  values_train,
                                  tensor,
                                  core_idx,
                                  num_cheb_pts_ptr,
                                  sorted_entries_ptr):
        """
        Implements alternating least squares scheme for the completion algorithm. Used to update each core of the TT tensor.

        For details we refer to 

        Steinlechner, M. Riemannian optimization for high-dimensional tensor completion. SIAM J. Sci. Comput. 38.  (2016). S461–S484.

        Parameters
        ----------
        tensor_entries_train : numpy array 2D
            Tensor subgrid entries used for training

        values_train : numpy array 1D
             Values on subgrid used for training

        tensor : TT
             Chebyshev Tensor in TT format.

        core_idx : int
             Indicates the core to be updated

        num_cheb_pts_ptr: ctype array with ints
            Number of chebyshev points per dimension
            Computed before calling this function to save time

        sorted_entries_ptr: dictionary of ctype array with ints


        Returns
        -------
        mod_core : numpy array 4D
             Core of TT tensor modified by ALS scheme

        """

        tensor_copy = copy.deepcopy(tensor)

        sorted_tensor_entries_train, sorted_values_train, __ = AlternatingLeastSquares.sort_tensor_pts(
            tensor_entries_train, values_train, core_idx)

        num_sorted_entries = sorted_tensor_entries_train.shape[0]

        ## -- Ranks
        ranks = np.array(tensor_copy.ranks, dtype=np.intc)
        ranks_ptr = ranks.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int * ranks.size))

        ## -- Results
        result_size = ranks[core_idx] * ranks[core_idx + 1]
        results = np.zeros(num_sorted_entries * result_size, dtype=np.double)
        results_ptr = results.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double * (num_sorted_entries * result_size)))

        dim = tensor_copy.order

        ## -- cores
        # t0 = time.time()
        cores = copy.deepcopy(tensor_copy.cores)
        modified_cores = [np.swapaxes(
            core[:, :, 0, :], 0, 1).flatten().tolist() for core in cores]
        cores_flat = np.concatenate(modified_cores)

        cores_ptr = cores_flat.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double * cores_flat.size))

        # -- New way of obtaining hom matrix. Calling wrapper
        hom_mat(sorted_entries_ptr, num_sorted_entries, ranks_ptr, num_cheb_pts_ptr, dim, core_idx,
                cores_ptr, results_ptr)

        hom_matrix_flat = np.ctypeslib.as_array(results_ptr.contents)
        hom_matrix = np.reshape(hom_matrix_flat,
                                (num_sorted_entries, ranks[core_idx] * ranks[core_idx + 1]))

        mod_core = AlternatingLeastSquares.__solve_lin_eqs(tensor_copy, core_idx,
                                                           sorted_tensor_entries_train,
                                                           sorted_values_train, hom_matrix)

        return mod_core

    def sort_tensor_pts(tensor_entries_train, values_train, core_idx):
        """
        Order tensor the subgrid of tensor used for training on the column corresponding to the core to update with ALS scheme. Used later for solving linear sys of equations

        Parameters
        ----------
        tensor_entries_train : numpy array 2D
            Tensor subgrid entries used for training

        values_train : numpy array 1D
             Values on subgrid used for training

        core_idx : int
             Indicates the core to be updated

        Returns
        -------
        sorted_tensor_entries_train : numpy array 2D
             The sorted subgrid (along column corresponding to core to updated) used for training

        sorted_values_train : numpy array 1D
             Sorted values corresponding to subgrid

        idx_sort : int
             Indices that give the sorting obtained

        """
        # get column to sort
        indices_core = tensor_entries_train[:, core_idx]
        # find indices that sort
        idx_sort = np.argsort(indices_core)

        # sort tensor points and tensor values using indices
        sorted_tensor_entries_train = tensor_entries_train[idx_sort, :]
        sorted_values_train = values_train[idx_sort]

        return sorted_tensor_entries_train, sorted_values_train, idx_sort

    @staticmethod
    def __solve_lin_eqs(tensor,
                        core_idx,
                        sorted_tensor_entries_train,
                        sorted_values_train,
                        hom_matrix):
        """
        Private method named as solve_lin_eqs.

        Parameters
        ----------
        tensor : TT
             Chebyshev Tensor in TT format.

        core_idx : int
             Indicates the core to be updated

        sorted_tensor_entries_train : numpy array 2D
             The sorted subgrid used for training

        sorted_values_train : numpy array 1D
             Sorted values corresponding to subgrid

        hom_matrix : numpy array 2D
             Obtained by wrapper of C code
             Used in system of linear equations

        Returns
        -------
        res : numpy array 4D
             Modified core

        """

        #print(f"Tensor inside of lin eqts solver: {tensor}")
        #res = np.empty((tensor.cores[core_idx].shape))
        res = copy.deepcopy(tensor.cores[core_idx])
        # slice_idx = 1
        for slice_idx in range(tensor.row_dims[core_idx]):
            bools = sorted_tensor_entries_train[:, core_idx] == slice_idx
            range_slice = np.where(bools == True)[0]

            sys_lin_A = hom_matrix[range_slice, :]
            sys_lin_B = sorted_values_train[range_slice]

            try:
                sol = np.linalg.lstsq(sys_lin_A, sys_lin_B, rcond=None)
            except LinAlgError as err:
                print(err)
            else:
                x = sol[0]

                core_slice_size = (
                    tensor.ranks[core_idx], tensor.ranks[core_idx + 1])
                core_slice = np.reshape(x, core_slice_size, order='F')
                res[:, slice_idx, 0, :] = core_slice

        return res
