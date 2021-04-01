import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from random import randint

import pickle
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error

from matplotlib import rc

plt.rc('text', usetex=True)
rc('text', usetex=True)
rc('xtick', labelsize=20) 
rc('ytick', labelsize=20)
