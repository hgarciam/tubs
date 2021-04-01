import numpy as np
import matplotlib.pyplot as plt

import pickle
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from matplotlib import rc

rc('text', usetex=False)
rc('xtick', labelsize=18)
rc('ytick', labelsize=18)

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

# Cleaning zeros

outliers_ID = []

IDnew = []
x1new = []
x2new = []
x3new = []
x4new = []
ynew = []

# Re-write more elegantly probably using anomaly detection algorithms.

for i in range(len(x1)):
    if x1[i] != 0 and x2[i] != 0 and x3[i] != 0 and x4[i] != 0 and y[i] != 0:
        x1new.append(x1[i])
        x2new.append(x2[i] - x1[i])
        # x2new.append(x2[i])
        x3new.append(x3[i])
        x4new.append(x4[i])
        ynew.append(y[i])
        IDnew.append(ID[i])
    else:
        # print('Sample with ID = %1.2i is out of standards' % ID[i])
        outliers_ID.append(ID[i])

print('First Clean Dataset length = %i ' % len(x1new))

meanX1 = np.mean(x1new)
stdX1 = np.std(x1new)
meanX2 = np.mean(x2new)
stdX2 = np.std(x2new)
meanX3 = np.mean(x3new)
stdX3 = np.std(x3new)
meanX4 = np.mean(x4new)
stdX4 = np.std(x4new)
meany = np.mean(ynew)
stdy = np.std(ynew)

ID = IDnew
x1 = scale(x1new)
x2 = scale(x2new)
x3 = scale(x3new)
x4 = scale(x4new)
y = scale(ynew)

###########################

IDnew = []
x1new = []
x2new = []
x3new = []
x4new = []
ynew = []

sigmaCutP = 3.0
sigmaCutN = -3.0

for i in range(len(x1)):
    if (sigmaCutP > x1[i] > sigmaCutN
            and sigmaCutP > x2[i] > sigmaCutN
            and sigmaCutP > x3[i] > sigmaCutN
            and sigmaCutP > x4[i] > sigmaCutN
            and sigmaCutP > y[i] > sigmaCutN
        ):
        x1new.append(x1[i])
        x2new.append(x2[i])
        x3new.append(x3[i])
        x4new.append(x4[i])
        ynew.append(y[i])
        IDnew.append(ID[i])
    else:
        # print('Sample with ID = %1.2i is out of standards' % ID[i])
        outliers_ID.append(ID[i])

# Rescale
x1 = scale(x1new)
x2 = scale(x2new)
x3 = scale(x3new)
x4 = scale(x4new)
y = scale(ynew)

X = np.c_[x1, x2, x3, x4]
T = np.c_[x1, x2, x3, x4, y]

print('Second Clean Dataset length = %i ' % len(x1new))
print('A total of %1.2i (%1.2f%%) samples have been discarted' % (
len(outliers_ID), len(outliers_ID) / len(data_df['IdContramostra']) * 100))

#############################

polyX = X

# Plot data matrix
dataPlot = False
if dataPlot:
    plt.figure(2)
    df = pd.DataFrame(T, columns=['x1', 'x2', 'x3', 'x4', 'y'])
    scatter_matrix(df, alpha=0.2, figsize=(12, 6), diagonal='hist')


# Training
X_train, X_test, y_train, y_test = train_test_split(polyX, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=60, random_state=0)
model.fit(X_train, y_train)

features = False
if features:
    # get importance
    importance = self.estimator.feature_importances_
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    plt.bar([x for x in range(len(importance))], importance)
    plt.show()

gridsearch = False
if gridsearch:
    param_grid = {
        #'bootstrap': [True],
        #'max_depth': [20],
        #'max_features': ['auto'],
        #'min_samples_leaf': [1],
        #'min_samples_split': [2, 3],
        'n_estimators': [30, 40, 50, 60]
    }

    gridsearch = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=1)
    gridsearch.fit(X_train, y_train)

    # Best cross validation score
    print('Cross Validation Score:', gridsearch.best_score_)
    # Best parameters which resulted in the best score
    print('Best Parameters:', gridsearch.best_params_)

print()
print("Random Forest")

errorTrain = model.score(X_train, y_train)
errorTest = model.score(X_test, y_test)

y_predict_train = model.predict(X_train)
y_predict_test = model.predict(X_test)

print(np.shape(y_predict_test))

print("Training set score: {:.3f}".format(model.score(X_train, y_train)))
print("MAE test: {:.3f}".format(mean_absolute_error(y_train, y_predict_train)))
print("Test set score: {:.3f}".format(model.score(X_test, y_test)))
print("MAE test: {:.3f}".format(mean_absolute_error(y_test, y_predict_test)))
print()

# Save models

PredTrain = model.predict(X_train) * stdy + meany
PredTest = model.predict(X_test) * stdy + meany
y_train_rescale = y_train * stdy + meany
y_test_rescale = y_test * stdy + meany
ResidualTrain = PredTrain - y_train_rescale
ResidualTest = PredTest - y_test_rescale

plt.figure()
plt.plot(y_train_rescale, ResidualTrain, '.', label='Train')
plt.plot(y_test_rescale, ResidualTest, '.', label='Test')
plt.xlabel('Resistencia', fontsize=18)
plt.ylabel('Prediction - True', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot1_ridge.pdf', bbox_inches='tight')

plt.figure()
plt.plot(y_train_rescale, ResidualTrain / y_train_rescale, '.', label='Train')
plt.plot(y_test_rescale, ResidualTest / y_test_rescale, '.', label='Test')
plt.xlabel('Resistencia', fontsize=18)
plt.ylabel('(Prediction - True)/True', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot2_ridge.pdf', bbox_inches='tight')

plt.figure()
plt.plot(y_train_rescale, PredTrain, '.', label='Train')
plt.plot(y_test_rescale, PredTest, '.', label='Test')
plt.xlabel('True', fontsize=18)
plt.ylabel('Prediction', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot3_ridge.pdf', bbox_inches='tight')

plt.figure()
plt.hist(PredTest, alpha=0.5, range=(0, 4000), label='Prediction')
plt.hist(y_test_rescale, alpha=0.5, range=(0, 4000), label='True')
plt.xlabel('Resistencia', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('plot4_ridge.pdf', bbox_inches='tight')

print("MAE train: {:.3f}".format(mean_absolute_error(y_train_rescale, PredTrain)))
print("MAE test: {:.3f}".format(mean_absolute_error(y_test_rescale, PredTest)))
plt.show()