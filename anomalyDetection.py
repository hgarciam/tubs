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

filename = 'Contramostres.xlsx'
data_df = pd.read_excel(filename)

ID = data_df['IdContramostra']
x1 = data_df['Interior']
x2 = data_df['Exterior']
x3 = data_df['Humitat']
x4 = data_df['Qualitat']
y = data_df['Resistencia']

print()
print('Original Dataset length = %i' % len(x1))

# Plot data matrix
dataPlot = True
if dataPlot == True:
    X = np.c_[x1,x2,x3,x4,y]
    plt.figure(2)
    df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4','y'])
    scatter_matrix(df, alpha=0.2, figsize=(12, 6), diagonal='hist')
    plt.show()
