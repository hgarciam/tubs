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

        else:
            self.ID = self.data_df['IdContramostra']
            self.x1 = self.data_df['Interior']
            self.x2 = self.data_df['Exterior']
            self.x3 = self.data_df['Humitat']
            self.x4 = self.data_df['Qualitat']
            self.y = self.data_df['Resistencia']

        print(self.data_df.head())

    def inout(self):
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
                x2new.append(self.x2[i] - self.x1[i])
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
        self.x1 = scale(x1new)
        self.x2 = scale(x2new)
        self.x3 = scale(x3new)
        self.x4 = scale(x4new)
        self.y = scale(ynew)

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

        sigmaCutP = 3.0
        sigmaCutN = -3.0

        for i in range(len(self.x1)):
            if (sigmaCutP > self.x1[i] > sigmaCutN
                    and sigmaCutP > self.x2[i] > sigmaCutN
                    and sigmaCutP > self.x3[i] > sigmaCutN
                    and sigmaCutP > self.x4[i] > sigmaCutN
                    and sigmaCutP > self.y[i] > sigmaCutN
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
        self.x1 = scale(x1new)
        self.x2 = scale(x2new)
        self.x3 = scale(x3new)
        self.x4 = scale(x4new)
        self.y = scale(ynew)

        x = np.c_[self.x1, self.x2, self.x3, self.x4]

        print('Second Clean Dataset length = %i ' % len(x1new))
        print('A total of %1.2i (%1.2f%%) samples have been removed' % (len(outliers_id), len(outliers_id) / len(self.id) * 100))

        return x, self.y
