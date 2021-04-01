import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
from pandas.plotting import scatter_matrix

import tubeclass

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


class Tubes:

	def __init__(self, datafile):
		self.tcl = tubeclass.Tubeclass(datafile)

		# Options
		self.CLEAN_ZEROS = True
		self.CLEAN_DATA = True
		self.PLOT_MATRIX = False
		self.LOAD_MODEL = True
		self.FEATURES = False
		self.CROSS_VAL = False
		self.GRID_SCAN = False
		self.MAKE_PLOTS = False

	def run(self):

		x, y = self.tcl.inout()

		meany = np.mean(y)
		stdy = np.std(y)

		if self.CLEAN_ZEROS:
			x, y = self.tcl.clean_zeros()

		if self.CLEAN_DATA:
			x, y = self.tcl.clean_data()

		# Plot data matrix
		if self.PLOT_MATRIX:
			t = np.c_[x, y]
			plt.figure()
			df = pd.DataFrame(t, columns=['x1', 'x2', 'x3', 'x4', 'y'])
			scatter_matrix(df, alpha=0.2, figsize=(12, 6), diagonal='hist')
			plt.show()

		print()
		print("Random Forest")

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

		filename = 'RF_model.pickle'

		if self.LOAD_MODEL:
			print('Loading old model...')
			model = pickle.load(open(filename, 'rb'))
		else:
			model = RandomForestRegressor(n_estimators=60, random_state=0)
			print('Fitting new model...')
			model.fit(x_train, y_train)

			pickle.dump(model, open(filename, 'wb'))

		if self.FEATURES:
			# get importance
			importance = model.feature_importances_
			for i, v in enumerate(importance):
				print('Feature: %0d, Score: %.5f' % (i, v))
			plt.figure()
			plt.bar([x for x in range(len(importance))], importance)
			plt.show()

		if self.GRID_SCAN:
			param_grid = {
				# 'bootstrap': [True],
				# 'max_depth': [20],
				# 'max_features': ['auto'],
				# 'min_samples_leaf': [1],
				# 'min_samples_split': [2, 3],
				'n_estimators': [40, 60, 70, 80]
			}

			gridsearch = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=1)
			gridsearch.fit(x_train, y_train)

			# Best cross validation score
			print('Cross Validation Score:', gridsearch.best_score_)
			# Best parameters which resulted in the best score
			print('Best Parameters:', gridsearch.best_params_)

		if self.CROSS_VAL:
			# Some cross-validat
			score = cross_val_score(model, x_train, y_train, cv=5)
			print("Cross validation: score = %1.4f +/- %1.4f" % (np.mean(score), np.std(score)))

		y_predict_train = model.predict(x_train)
		y_predict_test = model.predict(x_test)

		print("Training set score: {:.3f}".format(model.score(x_train, y_train)))
		print("MAE test: {:.3f}".format(mean_absolute_error(y_train, y_predict_train)))
		print("Test set score: {:.3f}".format(model.score(x_test, y_test)))
		print("MAE test: {:.3f}".format(mean_absolute_error(y_test, y_predict_test)))
		print()

		# Save models

		pred_train = model.predict(x_train) * stdy + meany
		pred_test = model.predict(x_test) * stdy + meany
		y_train_rescale = y_train * stdy + meany
		y_test_rescale = y_test * stdy + meany
		residual_train = pred_train - y_train_rescale
		residual_test = pred_test - y_test_rescale

		if self.MAKE_PLOTS:
			plt.figure()
			plt.plot(y_train_rescale, residual_train, '.', label='Train')
			plt.plot(y_test_rescale, residual_test, '.', label='Test')
			plt.xlabel('Resistencia', fontsize=18)
			plt.ylabel('Prediction - True', fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('plot1_RF.pdf', bbox_inches='tight')

			plt.figure()
			plt.plot(y_train_rescale, residual_train / y_train_rescale, '.', label='Train')
			plt.plot(y_test_rescale, residual_test / y_test_rescale, '.', label='Test')
			plt.xlabel('Resistencia', fontsize=18)
			plt.ylabel('(Prediction - True)/True', fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('plot2_RF.pdf', bbox_inches='tight')

			plt.figure()
			plt.plot(y_train_rescale, pred_train, '.', label='Train')
			plt.plot(y_test_rescale, pred_test, '.', label='Test')
			plt.xlabel('True', fontsize=18)
			plt.ylabel('Prediction', fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('plot3_RF.pdf', bbox_inches='tight')

			plt.figure()
			plt.hist(pred_test, alpha=0.5, range=(0, 4000), label='Prediction')
			plt.hist(y_test_rescale, alpha=0.5, range=(0, 4000), label='True')
			plt.xlabel('Resistencia', fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('plot4_RF.pdf', bbox_inches='tight')

			plt.show()


if __name__ == "__main__":
	#data = "Contramostres.xlsx"
	data = '2020_data/CONTRAMOSTRES_2019-2020_SP.xls'
	t0 = time()
	f = Tubes(data)
	f.run()
	print("Time required = %1.2f seconds" % float(time() - t0))