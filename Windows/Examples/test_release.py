
import numpy as np
import time
import sys
import os

import mocaxextendpy.mocax_extend as me
import black_scholes_utils as bs


np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


""" Function to approximate, dimension, domain and number of tensor points per dimension """

# -- Exponential function
def funct(x): return np.exp(-np.linalg.norm(x, axis=1))


dimension = 4
num_cheb_pts = dimension * [13]
variable_ranges = (dimension) * [[0, 1]]


# -- Black Scholes function
# def funct(x): return [bs.blackScholes("C", *pt) for pt in x]

# dimension = 5
# num_cheb_pts = dimension * [10]
# variable_ranges = [[50, 150], [50, 150], [0.01, 0.1], [0.2, 0.7], [1.0, 3.0]]


""" MocaxExtend object instantiation and subgrid creation"""

# -- Object instance
obj = me.MocaxExtend(dimension,
                     num_cheb_pts,
                     variable_ranges)


# -- Subgrid creation
num_scenarios = 1000
random_cheb_pts = obj.subgrid_by_number(num_scenarios)

# -- Function evaluation on subgrid
vals_subgrid = funct(random_cheb_pts)

# -- Incorporate data to rank adaptive object
obj.set_subgrid_values(vals_subgrid)
obj.gen_train_val_data()

# -- Size comparison of original grid with subgrid
original_grid_size = obj.get_tensor_size()
subgrid_size = obj.get_subgrid_size()

print("Subgrid only {:.5f}% of original grid\n".format(
    100 * subgrid_size / original_grid_size))

""" Rank adaptive algorithm run """

# -- Parameters defined
rank_adaptive_params = {"tolerance": 1e-3,
                        "rel_tolerance": 1e-8,
                        "max_iters": 100,
                        "max_rank": 10,
                        "print_progress": True,
                        "max_rounds": 5}


# -- Training of Chebyshev Tensor in TT format
obj.run_rank_adaptive_algo(**rank_adaptive_params)


""" Serialization and deserialization """

## -- Serialization
obj.serialize("tensor_extension_obj.pickle")

## -- Deserialization
obj_des = me.MocaxExtend.deserialize("tensor_extension_obj.pickle")


""" Evaluation of Chebyhsev Tensor in TT format on fine grid of points """

# -- Creation of fine grid
num_pts = 1000
random_grid = np.empty((num_pts, dimension))
for m, interval in enumerate(variable_ranges):
    #grid_interval = np.linspace(interval[0] + 1e-10, interval[1] - 1e-10)
    grid_interval = np.linspace(interval[0], interval[1])
    random_grid[:, m] = np.random.choice(grid_interval, size=num_pts)


# -- Evaluation of fine grid with function for bechmarking accuracy
vals_random_grid = funct(random_grid)


# -- Evaluation of fine grid with Chebyshev Tensor in TT format
t0 = time.perf_counter()
vals_cheb_tensor_random_grid = obj.cheb_tensor_evals(random_grid)
t1 = time.perf_counter()
total = t1 - t0
print("Chebyshev Tensor in TT format evaluation time on {} points: {:.5f} seconds\n".format(num_pts, total))


# -- Empirical error
diff = vals_random_grid - vals_cheb_tensor_random_grid
error_sup = max(abs(diff))
error_mean = abs(diff).mean()

print(f"Maximum error: {error_sup:.5f}")
print(f"Mean error: {error_mean:.5f}")
