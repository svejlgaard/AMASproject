import pandas as pd
import numpy as np
from sklearn import preprocessing
import os, contextlib, sys
from scipy import stats
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
        
    def MonteCarlo(self, MC_size):
        MC_list = list()
        for i in range(2):
            features = self.unscaled_features[self.targets == i]
            features = features.reset_index(drop=True)
            means = np.mean(features, axis=0)
            cov_matrix = features.corr()
            mc_features = stats.multivariate_normal(mean=means, cov=cov_matrix).rvs(MC_size, random_state=27)
            mc_features = pd.DataFrame(data=mc_features, columns = features.columns)
            mc_features['Class'] = np.ones(MC_size, dtype=int) * int(i)
            MC_list.append(mc_features)
        self.MC_data = pd.concat(MC_list).sample(frac=1).reset_index(drop=True)
        return self.MC_data