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

		''' Options '''
		self.CLEAN_ZEROS = True
		self.CLEAN_DATA = True
		self.PLOT_MATRIX = False
		self.LOAD_MODEL = False
		self.FEATURES = False
		self.CROSS_VAL = False
		self.GRID_SCAN = False
		self.MAKE_PLOTS = False
		self.GET_CLASSIC = True

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
			plt.xlabel('Feature')
			plt.ylabel('Relative importance')
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

		stdy = 1.
		meany = 0.
		pred_train = model.predict(x_train)
		pred_test = model.predict(x_test)
		y_train_rescale = y_train
		y_test_rescale = y_test
		residual_train = pred_train - y_train_rescale
		residual_test = pred_test - y_test_rescale

		print("MAE train: {:.3f}".format(mean_absolute_error(y_train_rescale, pred_train)))
		print("MAE test: {:.3f}".format(mean_absolute_error(y_test_rescale, pred_test)))

		if self.MAKE_PLOTS:
			plt.figure()
			plt.plot(y_train_rescale, residual_train, '.', label='Train')
			plt.plot(y_test_rescale, residual_test, '.', label='Test')
			plt.xlabel('Resistencia', fontsize=18)
			plt.ylabel('Prediction - True', fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('plot1_RF.pdf', bbox_inches='tight')

			plt.figure()
			plt.plot(y_train_rescale, residual_train / y_train_rescale*100, '.', label='Train')
			plt.plot(y_test_rescale, residual_test / y_test_rescale*100, '.', label='Test')
			plt.xlabel('Resistencia', fontsize=18)
			plt.ylabel('(Prediction - True)/True', fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('plot2_RF.pdf', bbox_inches='tight')

			plt.figure()
			#plt.hist(residual_train / y_train_rescale*100, range=(-50, 50), bins=20, alpha=0.5, label='Train')
			plt.hist(residual_test / y_test_rescale*100,  range=(-50, 50), bins=20, alpha=0.5, label='Test')
			plt.ylabel('Counts', fontsize=18)
			plt.xlabel('(Prediction - True)/True [%]', fontsize=18)
			plt.xticks(fontsize=18)
			plt.yticks(fontsize=18)
			plt.legend(fontsize=18)
			plt.savefig('hist_rel_diff_RF.pdf', bbox_inches='tight')

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

			plt.figure(10)
			plt.plot(pred_test[0:100], label='Prediction')
			plt.plot(y_test_rescale[0:100], label='True')
			plt.legend(fontsize=18)
			plt.ylabel('Resistance', fontsize=18)
			plt.savefig('plot5_RF.pdf', bbox_inches='tight')

			print(np.mean(residual_test / y_test_rescale)*100)
			print(np.std(residual_test / y_test_rescale)*100)

			reduced_prediction = [i for i in residual_test / y_test_rescale * 100 if abs(i) < 50.]
			print("Number samples with less than 50% error: " + str(len(reduced_prediction)))
			print("Mean of samples with less than 50% error: " + str(np.mean(reduced_prediction)))
			print("Standard deviation with less than 50% error: " + str(np.std(reduced_prediction)))

			res = sum(1 for i in residual_test/y_test_rescale*100 if abs(i) < 50.)
			print("Total number samples: " + str(len(residual_test)))
			print("Number samples with less than 50% error: " + str(res) + " (" + str(res/len(residual_test)*100) + "%)")

		x1, x2 = [], []
		for i in range(len(x_test)):
			if x_test[i,3] == 200:
				x1.append(x_test[i,0])
				x2.append(x_test[i,1])
				y.append(y_test_rescale[i])

		x1 = x_test[:, 0]
		x2 = x_test[:, 1]

		if self.GET_CLASSIC:

			classic1, classic2, classic3, classic4 = [], [], [], []

			for i in range(len(x1)):
				classic1.append(self.tcl.classic_estimator(200, x1[i], x2[i]))
				classic2.append(self.tcl.classic_estimator(250, x1[i], x2[i]))
				classic3.append(self.tcl.classic_estimator(400, x1[i], x2[i]))
				classic4.append(self.tcl.classic_estimator(600, x1[i], x2[i]))

			plt.figure(10)
			plt.plot(pred_test[0:100], label='Prediction')
			plt.plot(y_test_rescale[0:100], label='True')
			plt.plot(classic1[0:100], 'o', label='Classic 200')
			plt.plot(classic2[0:100], 'o', label='Classic 250 ')
			plt.plot(classic3[0:100], 'o', label='Classic 400')
			plt.plot(classic4[0:100], 'o', label='Classic 600')
			plt.legend(fontsize=18)
			plt.ylabel('Resistance', fontsize=18)

			plt.show()

			print("MAE C1 test: {:.3f}".format(mean_absolute_error(classic1, y_test_rescale)))
			print("MAE C2 test: {:.3f}".format(mean_absolute_error(classic2, y_test_rescale)))
			print("MAE C3 test: {:.3f}".format(mean_absolute_error(classic3, y_test_rescale)))
			print("MAE C4 test: {:.3f}".format(mean_absolute_error(classic4, y_test_rescale)))


if __name__ == "__main__":
	#data = "Contramostres.xlsx"
	data = '2020_data/CONTRAMOSTRES_2019-2020_SP.xls'
	t0 = time()
	f = Tubes(data)
	f.run()
	print("Time required = %1.2f seconds" % float(time() - t0))