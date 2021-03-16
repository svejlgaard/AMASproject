import pandas as pd
import os, contextlib, sys
# Sets the directory to the current directory
os.chdir(sys.path[0])

class PulsarData:
    '''A function to load the pulsar data.
    Return: pandas dataframe.'''
    def __init__(self, filename):
        self.filename=filename
        self.data = pd.read_csv(f'{self.filename}.csv', header=None, names=['Mean of the integrated profile.',
" Standard deviation of the integrated profile.",
'Excess kurtosis of the integrated profile.',
'Skewness of the integrated profile.',
'Mean of the DM-SNR curve.',
'Standard deviation of the DM-SNR curve.',
'Excess kurtosis of the DM-SNR curve.',
'Skewness of the DM-SNR curve.',
'Class'])

