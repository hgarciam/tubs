import numpy as np

import pandas as pd
from sklearn.preprocessing import scale


class Tubeclass(object):

    def __init__(self, data):
        self.data_df = pd.read_excel(data)

        dataset = 'new'

        if dataset == 'new':
            self.data_df['D.Ext'] = self.data_df['D.Ext'].str.replace(',', '.').astype(float)
            self.data_df['D.Int'] = self.data_df['D.Int'].str.replace(',', '.').astype(float)
            self.data_df['Humedad'] = self.data_df['Humedad'].str.replace(',', '.').astype(float)
            self.ID = self.data_df['OT']
            self.x1 = self.data_df['D.Int']
            self.x2 = self.data_df['D.Ext']
            self.x3 = self.data_df['Humedad']
            self.x4 = self.data_df['Calidad']
            self.y = self.data_df['Resistencia']

            self.mean_x1 = np.mean(self.x1)
            self.std_x1 = np.std(self.x1)
            self.mean_x2 = np.mean(self.x2)
            self.std_x2 = np.std(self.x2)

            print(self.mean_x1, self.std_x1, self.mean_x2, self.std_x2)

        else:
            self.ID = self.data_df['IdContramostra']
            self.x1 = self.data_df['Interior']
            self.x2 = self.data_df['Exterior']
            self.x3 = self.data_df['Humitat']
            self.x4 = self.data_df['Qualitat']
            self.y = self.data_df['Resistencia']

        print(self.data_df.head())

    def inout(self):
        self.x2 = (self.x2 - self.x1)
        x = np.c_[self.x1, self.x2, self.x3, self.x4]

        return x, self.y

    def clean_zeros(self):

        outliers_id = []

        idnew = []
        x1new = []
        x2new = []
        x3new = []
        x4new = []
        ynew = []

        # Re-write more elegantly probably using anomaly detection algorithms.

        for i in range(len(self.x1)):
            if self.x1[i] != 0 and self.x2[i] != 0 and self.x3[i] != 0 and self.x4[i] != 0 and self.y[i] != 0:
                x1new.append(self.x1[i])
                x2new.append(self.x2[i])
                # x2new.append(x2[i])
                x3new.append(self.x3[i])
                x4new.append(self.x4[i])
                ynew.append(self.y[i])
                idnew.append(self.ID[i])
            else:
                # print('Sample with ID = %1.2i is out of standards' % ID[i])
                outliers_id.append(self.ID[i])

        print('First Clean Dataset length = %i ' % len(x1new))

        self.id = idnew
        self.x1 = x1new
        self.x2 = x2new
        self.x3 = x3new
        self.x4 = x4new
        self.y = ynew

        x = np.c_[self.x1, self.x2, self.x3, self.x4]

        return x, self.y

    def clean_data(self):

        outliers_id = []
        idnew = []
        x1new = []
        x2new = []
        x3new = []
        x4new = []
        ynew = []

        x1norm = scale(self.x1)
        x2norm = scale(self.x2)
        x3norm = scale(self.x3)
        x4norm = scale(self.x4)
        ynorm = scale(self.y)

        sigmaCutP = 3.0
        sigmaCutN = -3.0

        for i in range(len(self.x1)):
            if (sigmaCutP > x1norm[i] > sigmaCutN
                    and sigmaCutP > x2norm[i] > sigmaCutN
                    and sigmaCutP > x3norm[i] > sigmaCutN
                    and sigmaCutP > x4norm[i] > sigmaCutN
                    and sigmaCutP > ynorm[i] > sigmaCutN
            ):
                x1new.append(self.x1[i])
                x2new.append(self.x2[i])
                x3new.append(self.x3[i])
                x4new.append(self.x4[i])
                ynew.append(self.y[i])
                idnew.append(self.id[i])
            else:
                # print('Sample with ID = %1.2i is out of standards' % ID[i])
                outliers_id.append(self.id[i])

        # Rescale
        self.x1 = x1new
        self.x2 = x2new
        self.x3 = x3new
        self.x4 = x4new
        self.y = ynew

        x = np.c_[self.x1, self.x2, self.x3, self.x4]

        print('Second Clean Dataset length = %i ' % len(x1new))
        print('A total of %1.2i (%1.2f%%) samples have been removed' % (len(outliers_id), len(outliers_id) / len(self.id) * 100))

        return x, self.y

    def classic_estimator(self, pb, x1, x2):

        if pb == 200:
            estimation = 104.+(136.*x2/2.)-(8.5*(x1/2.))
        elif pb == 250:
            estimation = 119.+(154.*x2/2.)-(8.41*(x1/2.))
        elif pb == 400:
            estimation = 139.+(178.*x2/2.)-(8.72*(x1/2.))
        else:
            estimation = 150.+(190.*x2/2.)-(5.2*(x1/2.))

        return estimation
