# mocaxextendpy - Python implementation of Tensor Extension Algorithms
# Copyright 2021 iRuiz Technologies, ltd.
# See LICENSE.txt for details on the End User License Agreement.

""" script defines class TTChebyshevUtils"""

import copy
import scipy
import numpy as np
from scikit_tt.tensor_train import TT


class TTChebyshevUtils:
    """Class that manages many of the functions used in training and evalution of Chebyhev Tensors in TT format."""

    @staticmethod
    def domain_scaler(a, b, pt):
        """
        Takes pt within inteval [a, b] and scales it to interval [-1,1]

        Parameters
        ----------
        a : float
             Determines starting point of interval [a, b]

        b : float
            Determines end point of interval [a, b]

        pt : float
             pt to scale

        Returns
        -------
        val : float
             scaled point

        Raises
        ------
        ValueError :
             The pt should be greater than a and less than b
             The value a should be less than value b.

        """
        if a < b:
            # if a < pt < b:
            if a <= pt and pt <= b:
                m = 2 / (b - a)
                c = (b + a) / (a - b)
                val = m * pt + c
                return val
            else:
                raise ValueError(
                    "The pt {} should be greater than {} and less than {}".format(pt, a, b))
        else:
            raise ValueError(
                "The value {} should be less than value {}".format(a, b))

    @staticmethod
    def c_tensor(tensor, num_cheb_pts):
        """
         This method builds the tensor of Chebyshev coefficients needed in the evaluation of the Chebyshev tensors in TT format. It is one of the two tensors in TT format needed for the inner product that implements evaluation of Chebyshev tensors in TT format.

         Parameters
         ----------
         tensor : TT
             tensor in TT format.

        num_cheb_pts : list of ints
             Number of Chebyshev points in each dimension

         Returns
         -------
         c_tensor : TT
             Chebyshev coefficients needed for evaluation

         Raises
         ------
         TypeError : type mismatch
             The tensor is not a tensor in TT format.

         """

        if isinstance(tensor, TT):
            tensor_copy = copy.deepcopy(tensor)

            f_mat = TTChebyshevUtils.__f_matrix(num_cheb_pts[0])
            c_tensor = TTChebyshevUtils.__m_mod_mult(tensor_copy, f_mat, 0)

            for m in range(1, len(num_cheb_pts)):
                f_mat = TTChebyshevUtils.__f_matrix(num_cheb_pts[
                    m])  # check whats the n vale of cheb points that needs to be passed. Remember that in pyhin n gives n+1 cheb pts
                c_tensor = TTChebyshevUtils.__m_mod_mult(c_tensor, f_mat, m)
            return c_tensor
        else:
            raise TypeError("The tensor is not a tensor in TT format")

    @staticmethod
    def __f_matrix(n):
        """
        Builds matrix used in mu matrix multiplication implemented in __m_mod_mult.

        Parameters
        ----------
        n : int
             It is 0 or greater than 0,Param passed is n but the size of matrix returned is (n+1) * (n+1)

        Returns
        -------
        fmat : numpy array
             It is F matrix.

        Raises
        ------
        TypeError : type mismatch
            n must be an integer
        ValueError: 
            n must be greater than 0

        """
        if isinstance(n, int):
            if n > 0:
                f_mat = np.empty((n + 1, n + 1))
                for j in range(n + 1):
                    for k in range(n + 1):
                        [cos_part, k_part, j_part] = TTChebyshevUtils.__f_matrix_entry(
                            n, j, k)
                        f_jk = j_part * k_part * cos_part
                        f_mat[j, k] = f_jk

                return f_mat
            else:
                raise ValueError(
                    "The n should be greater than 0.")
        else:
            raise TypeError("The n must be integer type")

    @staticmethod
    def __f_matrix_entry(n, j, k):
        """
        Computes value to fill matrix constructed in __f_matrix.

        Parameters
        ----------
        n : int

        j : int

        k : int

        Returns
        -------
        cos_part : ndarray
             The corresponding cosine values.

        k_part : float
             It returns float k_part.

        j_part : int
             It returns int j_part.

        """

        cos_part = np.cos(j * np.pi * k / n)
        if j == 0 or j == n:
            j_part = 1 / n
        else:
            j_part = 2 / n

        if k == 0 or k == n:
            k_part = 0.5
        else:
            k_part = 1

        return cos_part, k_part, j_part

    @staticmethod
    def __m_mod_mult(tensor, f_mat, m):
        """
        Implements m-mod multiplifaciton of tensor in TT tensor and f_mat. Used in the creation of the tensor c_tensor which is needed in the ealuation of Chebyshev Tensors in TT format

        Parameters
        ----------
        tensor : TT
             tensor in TT format.

        f_mat : numpy array
             It is F matrix.

        m : int
             It is 0 or greater than 0.

        Returns
        -------
        tensor : TT
             It returns tensor in TT format.

        Raises
        ------
        ValueError : size mismatch
             The m should be 0 or greater and integer.

        TypeError : type mismatch
             The tensor is not a tensor in TT format.
             m should be an integer

        """
        if isinstance(tensor, TT):
            if m >= 0:
                if isinstance(m, int):
                    core = tensor.cores[m]
                    core_prod = TTChebyshevUtils.__two_mod_mult(core, f_mat)
                    tensor.cores[m] = core_prod
                    return tensor
                else:
                    raise TypeError("The m should be integer")
            else:
                raise ValueError("The m should be 0 or greater.")
        else:
            raise TypeError("The tensor is not a in TT format")

    @staticmethod
    def __two_mod_mult(core, f_mat):
        """
        Implements 2-mod multiplication. This is needed in mu mod multiplcation.

        Parameters
        ----------
        core : numpy array 4D
             It is the cores of a TT tensor.

        f_mat : numpy array
             It is F matrix.

        Returns
        -------
        result : numpy array 4D
             It is the core that results from two mod multipliation

        Raises
        ------
        ValueError : size mismatch
             The core should be a 4D numpy array.

        """
        if core.ndim == 4:
            core_dims = np.shape(core)
            n_rows = core_dims[0]
            n_cols = core_dims[1]
            n_layers = core_dims[3]

            result = np.empty((n_rows, n_cols, 1, n_layers))
            for i in range(n_rows):
                for j in range(n_cols):
                    for k in range(n_layers):
                        core_slice = core[i, :, 0, k]
                        f_slice = f_mat[j, :]
                        entry = np.dot(core_slice, f_slice)
                        result[i, j, 0, k] = entry
            return result
        else:
            raise ValueError("The core should be a 4D numpy array")

    @staticmethod
    def gen_cheb_tensor(pt, dim, num_cheb_pts):
        """
        This method genrates the rank TT tensor obtained by evaluating Chebyshev polynomials on the point to be evaluated by a Chebyshev tensor in TT format. It is one of the two tensors in TT format needed for the inner product that implements evaluation of Chebyshev tensors in TT format.

        Parameters
        ----------
        pt : numpy array 1D
             point to be evaluated by Chebyshev Tensor in TT format

        dim : int
             Dimension of tensor

        num_cheb_pts : list
             List with number of chebyshev points per dimension

        Returns
        -------
        tensor : TT
             Chebyshev Tensor in TT format

        Raises
        ------
        ValueError : size mismatch
             pt must have the same number of entries as dimension of tensor
             there must be as many entries in num_cheb_pts as dimension of tensor
        """
        if dim == pt.size:
            if len(num_cheb_pts) == dim:
                cores = []
                for i in range(dim):
                    pt_1d = pt[i]
                    core_simple = TTChebyshevUtils.__cheb_funct(
                        pt_1d, num_cheb_pts[i])
                    core = core_simple.reshape(1, len(core_simple), 1, 1)
                    cores.append(core)
                tensor = TT(cores)
                return tensor
            else:
                raise ValueError(
                    "The length of num_cheb_pts must be the same as the dimension of tensor.")
        else:
            raise ValueError(
                "The dimension of pt must be the same as the dimension specified in the dimension of tensor.")

    @staticmethod
    def __cheb_funct(pt, n):
        """
        Evaluats Chebyshev polynomials on the point to be evaluated by a Chebyshev tensor in TT format. This method is used by gen_cheb_tensor.

        Parameters
        ----------
        pt : numpy array 1D
             point to be evaluated by Chebyshev Tensor in TT format

        n : int
             n is int type.
        Returns
        -------
        core : numpy array

        Raises
        ------
        ValueError : size mismatch
             n mush be greater than 0.

        TypeError : type mismatch
             pt must be of np.float and n must be of Integer type.

        """
        if isinstance(pt, np.float64):
            if isinstance(n, int):
                if n > 0:
                    k = n + 1  # Adjust to get right number of values
                    # list with zeros to determine which cheb pol we want
                    c = k * [0]
                    core = np.empty((k))  # values of core to return

                    for idx in range(k):  # Loop through degrees of cheb pols
                        c[idx] = 1  # Set the cheb pol for this iteration
                        core[idx] = np.polynomial.chebyshev.chebval(
                            pt, c)  # eval cheb pol on pt
                        c[idx] = 0  # reset cheb pol chosen for next iteration

                    return core
                else:
                    raise ValueError("n must be of greater than zero.")
            else:
                raise TypeError("n must be of Integer type.")
        else:
            raise TypeError("pt must be of np.float64.")

    @staticmethod
    def tt_tensor_inner_prod(tensor_1, tensor_2):
        """
        This method implements the inner product of two tensors in TT format.

        Parameters
        ----------
        tensor_1 : TT
             tensor in TT format.

        tensor_2 : TT
             tensor in TT format.

        Returns
        -------
        res : numpy array
             It returns res.

        Raises
        ------
        ValueError : size mismatch
             The dimension of tensor_1 must be the same as the dimension of tensor_2 .

        TypeError : type mismatch
             tensor_1 and tensor_2 must be of type TT tensor.

        """
        # Checking tensors are of TT format type
        if isinstance(tensor_1, TT):
            if isinstance(tensor_2, TT):

                # Checking the dimensions of the tensors passed is the same
                if tensor_1.order == tensor_2.order:

                    cores_1 = tensor_1.cores
                    cores_2 = tensor_2.cores

                    dim = tensor_1.order
                    core_1 = cores_1[0][:, :, 0, :]
                    core_2 = cores_2[0][:, :, 0, :]

                    core_1_unf = TTChebyshevUtils.__unfold_left(core_1)
                    core_2_unf = TTChebyshevUtils.__unfold_left(core_2)

                    res = np.dot(core_1_unf.transpose(), core_2_unf)

                    for i in range(1, dim):
                        core_1 = cores_1[i][:, :, 0, :]
                        core_2 = cores_2[i][:, :, 0, :]

                        tmp = TTChebyshevUtils.__tensor_prod(
                            core_1, res.transpose(), 1)
                        tmp_unf = TTChebyshevUtils.__unfold_left(tmp)
                        core_2_unf = TTChebyshevUtils.__unfold_left(core_2)

                        res = np.dot(tmp_unf.transpose(), core_2_unf)

                    return res
                else:
                    raise ValueError(
                        "The dimension of tensor_1 {} must be the same as the dimension of tensor_2 {}.".format(tensor_1.order, tensor_2.order))
            else:
                raise TypeError("tensor_2 must be of type TT tensor.")
        else:
            raise TypeError("tensor_1 must be of type TT tensor.")

    @staticmethod
    def __unfold_left(core):
        """
        Left unfolding of a core.

        Parameters
        ----------
        core : numpy array 3D
             Core to be unfolded (left)

        Returns
        -------
        core_reshaped : numpy array
             Unfolded core

        Raises
        ------
        ValueError : size mismatch
             The dimension of core is not equal to 3.
             This method works under the assumption that the core has dimension 3

        """
        if core.ndim == 3:
            core_shape = core.shape
            core_reshaped = core.reshape(
                (core_shape[0] * core_shape[1], core_shape[2]), order='F')
            return core_reshaped
        else:
            raise ValueError("The dimension of core is not equal to 3")

    @staticmethod
    def __tensor_prod(core, matrix, mode):
        """
        Multiplication of a core (3d array) with a matrix (2d array)

        Parameters
        ----------
        core : numpy array 3D

        matrix : numpy array 2D

        mode : int
             left right mode must be integer.

        Returns
        -------
        res : numpy array
             Result of performing tensor product

        """
        d = list(core.shape)
        d[mode - 1] = matrix.shape[0]
        res = np.dot(matrix, TTChebyshevUtils.__matricize(core, mode))
        res = TTChebyshevUtils.__tensorize(res, mode, d)
        return res

    @staticmethod
    def __matricize(core, mode):
        """
        Takes a core and turns it into a matrix.
        Only mode 1 and 3 supported

        Parameters
        ----------
        core : numpy array 3D
            Core to be matricised

        mode : int
            only mode 1 and 3 suppoted

        Returns
        -------
        res : numpy array 2D
            Matricised core

        Raises
        ------
        ValueError : size mismatch
             The dimension of core is not equal to 3.
             Mode has to be either 1 or 3

        TypeError : type mismatch
             mode must be of type Integer.

        """
        if core.ndim == 3:
            if isinstance(mode, int):
                d = core.shape
                if mode == 1:
                    res = core.reshape([d[0], d[1] * d[2]], order='F')
                    return res
                elif mode == 3:
                    core_reshaped = core.reshape(
                        [d[0] * d[1], d[2]], order='F')
                    res = core_reshaped.transpose()
                    return res
                else:
                    raise ValueError(
                        "Invalid mode input in function matricize.")
            else:
                raise TypeError("The mode must be of type Integer.")
        else:
            raise ValueError("The dimension of core is not equal to 3")

    @staticmethod
    def __tensorize(matrix, mode, d):
        """
        Takes a matrix and turns it into a core. It is the inverse to matricize.
        Modes 1 and 3 are the only ones supported

        Parameters
        ----------
        matrix : numpy array 2D
            matrix to be tensorized

        mode : int
            mode must be either 1 or 3

        d: int
            shape of tensorized matrix

        Returns
        -------
        res : numpy array 4D
             It returns res.

        Raises
        ------
        ValueError : size mismatch
             The dimension of matrix is not 2.

        TypeError : type mismatch
             Parameter mode must be an integer.

        ValueError : mode mismatch
             Invalid mode input in function tensorize. It should be either 1 or 3.

        """
        if matrix.ndim == 2:
            if isinstance(mode, int):
                if mode == 1:
                    res = matrix.reshape(d, order='F')
                    return res
                elif mode == 3:
                    mat_trans = matrix.transpose()
                    res = np.reshape(mat_trans, d, order='F')
                    return res
                else:
                    raise ValueError(
                        "Invalid mode input in function tensorize. It should be either 1 or 3")
            else:
                raise TypeError("Parameter mode must be an integer.")
        else:
            raise ValueError("The dimension of matrix is not 2")

    @staticmethod
    def orthogonalize(tensor, position):
        """
        Orthogonolizes a tensor in TT format at the core indicated by position.

        Parameters
        ----------
        tensor : TT
             tensor to be orthogonlized

        position : int
             Core to which we do the orthogonlization
             Should be between 0 and dimension - 1 (inclusive)

        Returns
        -------
        tt_tensor : TT
             TT tensor that has been orthogonalized

         Raises
        ------
        ValueError : size mismatch
             Position should be either 0 or more and less than or equal to dimension -1

        TypeError : type mismatch
             Parameter position must be an integer.
             tensor should be of TT type
        """

        if isinstance(tensor, TT):
            tt_tensor = copy.deepcopy(tensor)
            if isinstance(position, int):
                if 0 <= position < tt_tensor.order:

                    for i in range(position - 1):
                        tt_tensor = TTChebyshevUtils.orth_left(tt_tensor, i)
                    for i in range(tt_tensor.order - 1, position, -1):
                        tt_tensor = TTChebyshevUtils.orth_right(tt_tensor, i)
                    return tt_tensor

                else:
                    raise ValueError(
                        "Parameter position should be and greater than or equal to 0 and smaller or equal to {}".format(tensor.order - 1))
            else:
                raise TypeError("Position must be an integer")
        else:
            raise TypeError("Type of tensor must be TT.")

    @staticmethod
    def orth_left(tensor, position):
        """
        Orthogonalizes the full tensor.till position starting from the left.

        Parameters
        ----------
        tensor : TT
            Tensor to be orthogonlized

        position : int
             Core to which we do the orthogonlization
             Should be between 0 and dimension - 1 (inclusive)

        Returns
        -------
        tensor : TT
             TT tensor that has been orthogonalized

        Raises
        ------

        ValueError : size mismatch
             Position should be either 0 or more and less than or equal to dimension -1

        TypeError : type mismatch
             Parameter position must be an integer.
             tensor should be of TT type
        """
        if isinstance(tensor, TT):
            tt_tensor = copy.deepcopy(tensor)
            if isinstance(position, int):
                if 0 <= position < tt_tensor.order - 1:

                    sz = np.shape(tt_tensor.cores[position])
                    new_core = np.squeeze(tt_tensor.cores[position], axis=2)
                    q, r = np.linalg.qr(
                        TTChebyshevUtils.__unfold_left(new_core))
                    tt_tensor.cores[position] = np.expand_dims(q.reshape([sz[0], sz[1],
                                                                          q.shape[1]], order='F'), axis=2)
                    new_position_core = np.squeeze(
                        tt_tensor.cores[position + 1], axis=2)
                    tt_tensor.cores[position + 1] = np.expand_dims(TTChebyshevUtils.__tensor_prod(
                        new_position_core, r, 1), axis=2)
                    return tt_tensor
                else:
                    raise Exception(
                        "Position when calling orth_left should be and greater than or equal to 0 and smaller or equal to {}".format(
                            tensor.order - 2))
            else:
                raise TypeError("Position must be an Integer")
        else:
            raise TypeError("Type of tensor must be TT.")

    @staticmethod
    def orth_right(tensor, position):
        """
        Orthogonalizes the full tensor.till position starting from the right.

        Parameters
        ----------
        tensor : TT
            Tensor to be orthogonlized

        position : int
             Core to which we do the orthogonlization
             Should be between 0 and dimension - 1 (inclusive)

        Returns
        -------
        tensor : TT
             TT tensor that has been orthogonalized

        Raises
        ------
        ValueError : size mismatch
             Position should be either 0 or more and less than or equal to dimension -1

        TypeError : type mismatch
             Parameter position must be an integer.
             tensor should be of TT type
        """

        if isinstance(tensor, TT):
            tt_tensor = copy.deepcopy(tensor)
            if isinstance(position, int):
                if 0 < position < tt_tensor.order:

                    sz = np.shape(tt_tensor.cores[position])
                    #print(f"Size {sz}")
                    new_core = np.squeeze(tt_tensor.cores[position], axis=2)
                    #print(f"new core shape {new_core.shape}")

                    # q, r = np.linalg.qr(TTChebyshevUtils.__unfold_right(new_core))
                    A = TTChebyshevUtils.__unfold_right(new_core).transpose()
                    #print(f"shape of A: {A.shape}")
                    q, r = scipy.linalg.qr(A, mode='economic')
                    #print(f"Sahpe of r: {r.shape}")

                    q_trans = q.transpose()
                    tt_tensor.cores[position] = np.expand_dims(q_trans.reshape([q.shape[1], sz[1],
                                                                                sz[3]], order='F'), axis=2)
                    new_position_core = np.squeeze(
                        tt_tensor.cores[position - 1], axis=2)
                    #print(f"new core shape {new_position_core.shape}")

                    #print(f"Before: {tt_tensor.cores[3].shape}\n")
                    tt_tensor.cores[position - 1] = np.expand_dims(TTChebyshevUtils.__tensor_prod(
                        new_position_core, r, 3), axis=2)
                    #print(f"after: {tt_tensor.cores[3].shape}\n")

                    return tt_tensor
                else:
                    raise Exception(
                        "Position when calling orth_right should be strictly greater than 0 and smaller or equal than {}".format(
                            tensor.order - 1))
            else:
                raise TypeError("Position must be an Integer")
        else:
            raise TypeError("Type of tensor must be TT.")

    @staticmethod
    def __unfold_right(core):
        """
        Right unfolding of a core.

        Parameters
        ----------
        core : numpy array 3D
             Core to be unfolded

        Returns
        -------
        core_reshaped : numpy array 2D
             Unfolded core

        Raises
        ------
        ValueError : size mismatch
             The dimension of core is not equal to 3.

        """
        if core.ndim == 3:
            core_shape = core.shape
            core_reshaped = core.reshape(
                (core_shape[0], core_shape[1] * core_shape[2]), order='F')
            return core_reshaped
        else:
            raise ValueError("The dimension of core is not equal to 3")
