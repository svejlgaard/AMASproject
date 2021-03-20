import pandas as pd
import numpy as np
from sklearn import preprocessing
import os, contextlib, sys
# Sets the directory to the current directory
os.chdir(sys.path[0])

class PulsarData:
    '''
    A function to load the pulsar data.
    Return: pandas dataframe.
    '''
    def __init__(self, filename):
        self.filename=filename
        self.data = pd.read_csv(f'{self.filename}.csv', header=None, names=['Mean of the integrated profile',
'Standard deviation of the integrated profile',
'Excess kurtosis of the integrated profile',
'Skewness of the integrated profile',
'Mean of the DM-SNR curve',
'Standard deviation of the DM-SNR curve',
'Excess kurtosis of the DM-SNR curve',
'Skewness of the DM-SNR curve',
'Class'])
        self.targets = self.data['Class']
        self.unscaled_features = self.data.drop(columns='Class')

        scaler = preprocessing.MinMaxScaler()
        self.features = pd.DataFrame(scaler.fit_transform(self.unscaled_features), columns=self.unscaled_features.columns)


        # Adding a baseline function
        if len(np.unique(self.targets)) == 2:
            zeros = self.targets[self.targets == 0]
            ones = self.targets[self.targets == 1]
            len_zeros = len(zeros) / len(self.targets)
            len_ones = len(ones) / len(self.targets)
            self.baseline = len_zeros
            #print(f'The data set contains {len_zeros*100:.1f} % pulsars of class 0 and {len_ones*100:.1f} % pulsars of class 1')

