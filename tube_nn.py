import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from random import randint

import pickle
import pandas as pd
from pandas.plotting import scatter_matrix

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, r2_score

from matplotlib import rc

plt.rc('text', usetex=True)
rc('text', usetex=True)
rc('xtick', labelsize=18) 
rc('ytick', labelsize=18)

filename = 'Contramostres.xlsx'
data_df = pd.read_excel(filename)

print(data_df.head())

x1 = data_df['Interior']
x2 = data_df['Exterior']
x3 = data_df['Humitat']
x4 = data_df['Qualitat']
y = data_df['Resistencia']

print('Original Dataset length = %i' % len(x1))

# Cleaning zeros

x1new = []
x2new = []
x3new = []
x4new = []
ynew = []

for i in range(len(x1)):
    if (x1[i] != 0 and x2[i] != 0 and x3[i] != 0 and x4[i] != 0 and y[i] != 0):
        x1new.append(x1[i])
        x2new.append(x2[i]-x1[i]) # Redefine radius to obtain wall width
        #x2new.append(x2[i])
        x3new.append(x3[i])                                                                  
        x4new.append(x4[i])
        ynew.append(y[i])  

print('Dataset length after cleaning zeros = %i ' % len(x1new))

meanX1 = np.mean(x1new)
stdX1 = np.mean(x1new)
meanX2 = np.mean(x2new)
stdX2 = np.mean(x2new)
meanX3 = np.mean(x3new)
stdX3 = np.mean(x3new)
meanX4 = np.mean(x4new)
stdX4 = np.mean(x4new)
meany = np.mean(ynew)
stdy = np.mean(ynew)

x1 = scale(x1new)
x2 = scale(x2new)
x3 = scale(x3new)
x4 = scale(x4new)
y = scale(ynew)

###########################

x1new = []
x2new = []
x3new = []
x4new = []
ynew = []

sigmaCutP = 3.0
sigmaCutN = -3.0

for i in range(len(x1)):
    if (x1[i] < sigmaCutP and x1[i] > sigmaCutN \
    and x2[i] < sigmaCutP and x2[i] > sigmaCutN \
    and x3[i] < sigmaCutP and x3[i] > sigmaCutN \
    and x4[i] < sigmaCutP and x4[i] > sigmaCutN \
	and y[i] < sigmaCutP and y[i] > sigmaCutN \
	):
        x1new.append(x1[i])
        x2new.append(x2[i])
        x3new.append(x3[i])                                                                  
        x4new.append(x4[i])
        ynew.append(y[i])  

print('Second Clean Dataset length = %i ' % len(x1new))

# Rescale
x1 = scale(x1new)
x2 = scale(x2new)
x3 = scale(x3new)
x4 = scale(x4new)
y = scale(ynew)

#############################

# Normalization
X = np.c_[x1,x2,x3,x4]

# Create Polynomial Features
poly = PolynomialFeatures(1)
polyX = poly.fit_transform(X)

T = np.c_[x1,x2,x3,x4,y]

X_train, X_test, y_train, y_test = train_test_split(polyX, y, test_size=0.2)

###############################################################################

model1 = keras.Sequential([
    layers.BatchNormalization(input_shape=[5]),
	layers.Dense(units=50, activation='relu'),
    layers.BatchNormalization(),
	layers.Dense(units=10, activation='relu'),
    layers.BatchNormalization(),
	layers.Dense(units=1, activation='linear'),
])


es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=200)

model1.compile(optimizer='adam',loss='mae')
history = model1.fit(X_train, y_train, epochs=500, batch_size=256, validation_data=(X_test,y_test),callbacks=[es])
history_df = pd.DataFrame(history.history)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')

PredTrain = model1.predict(X_train)*stdy+meany
PredTest = model1.predict(X_test)*stdy+meany
y_train_rescale = y_train*stdy+meany
y_test_rescale = y_test*stdy+meany
ResidualTrain = PredTrain[:,0] - y_train_rescale
ResidualTest = PredTest[:,0] - y_test_rescale

plt.figure()
plt.plot(y_train_rescale,ResidualTrain,'.',label='Train')
plt.plot(y_test_rescale,ResidualTest,'.',label='Test')
plt.xlabel('Resistencia',fontsize=18)
plt.ylabel('Prediction - True',fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot1_NN.pdf',bbox_inches='tight')

plt.figure()
plt.plot(y_train_rescale,ResidualTrain/y_train_rescale,'.',label='Train')
plt.plot(y_test_rescale,ResidualTest/y_test_rescale,'.',label='Test')
plt.xlabel('Resistencia',fontsize=18)
plt.ylabel('(Prediction - True)/True',fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot2_NN.pdf',bbox_inches='tight')

plt.figure()
plt.plot(y_train_rescale,PredTrain,'.',label='Train')
plt.plot(y_test_rescale,PredTest,'.',label='Test')
plt.xlabel('True',fontsize=18)
plt.ylabel('Prediction',fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot3_NN.pdf',bbox_inches='tight')

plt.figure()
plt.hist(PredTest,alpha=0.5,range=(0,4000),label='Prediction')
plt.hist(y_test_rescale,alpha=0.5,range=(0,4000),label='True')
plt.xlabel('Resistencia',fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot4_NN.pdf',bbox_inches='tight')

print('R2 score Train =', r2_score(PredTrain,y_train_rescale))
print('R2 score Test =', r2_score(PredTest,y_test_rescale))
print("MAE train: {:.3f}".format(mean_absolute_error(y_train_rescale, PredTrain)))
print("MAE test: {:.3f}".format(mean_absolute_error(y_test_rescale, PredTest)))
