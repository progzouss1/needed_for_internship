# mocaxextendpy - Python implementation of Tensor Extension Algorithms
# Copyright 2021 iRuiz Technologies, ltd.
# See LICENSE.txt for details on the End User License Agreement.

"""The MocaxextendAlgorithm C interface for Python."""

import sys
import platform
import ctypes

if platform.system() == "Windows":
    tensorvals_lib_filename = r"tensorvals.dll"
    hom_mat_lib_filename = r"hommat.dll"

else:
    tensorvals_lib_filename = r"libtensorvals.so"
    hom_mat_lib_filename = r"libhommat.so"

# Name of the library loaded
tensor_vals_lib = ctypes.cdll.LoadLibrary(tensorvals_lib_filename)
hom_mat_lib = ctypes.cdll.LoadLibrary(hom_mat_lib_filename)

hom_mat_lib.argtypes = (ctypes.POINTER(ctypes.c_int),
                        int,
                        ctypes.POINTER(ctypes.c_int),
                        ctypes.POINTER(ctypes.c_int),
                        int,
                        int,
                        ctypes.POINTER(ctypes.c_double),
                        ctypes.POINTER(ctypes.c_double))

hom_mat_lib.restype = None


tensor_vals_lib.argtypes = (ctypes.POINTER(ctypes.c_int),
                            int,
                            ctypes.POINTER(ctypes.c_int),
                            ctypes.POINTER(ctypes.c_int),
                            int,
                            ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double))

tensor_vals_lib.restype = None


# Python function for value recovery of tensor grid
def tensor_vals(indices_array, num_indices, ranks_array, num_cheb_pts_array, dim, cores_array,
                results_array):
    """
    Method for calling so file of tensor_vals c file
    """
    return tensor_vals_lib.tensor_vals(indices_array, num_indices, ranks_array,
                                       num_cheb_pts_array, dim, cores_array, results_array)


# Python function to obtain the homogenous matrix in the ALS scheme
def hom_mat(indices_array, num_indices, ranks_array, num_cheb_pts_array, dim,
            core_idx, cores_array, results_array):
    """
    Method for calling so file of hom_mat c file
    """
    return hom_mat_lib.hom_matrix(indices_array, num_indices, ranks_array, num_cheb_pts_array, dim, core_idx,
                                  cores_array, results_array)
