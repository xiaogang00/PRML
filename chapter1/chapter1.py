import numpy as np   
import matplotlib.pyplot as plt           
from prml.features import PolynomialFeatures
from prml.linear import (
    LinearRegressor,
    RidgeRegressor,
    BayesianRegressor
)


np.random.seed(1234)

## Example: polynomial curve fitting
