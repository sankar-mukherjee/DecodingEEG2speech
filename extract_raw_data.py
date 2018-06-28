import mne
import scipy.io
#import listen_italian_functions
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import warnings
import msvcrt as m
import pickle


def wait():
    m.getch()

warnings.filterwarnings('ignore')

data_path = "./coherence_epochs/"

subject_name = ['Alice','Andrea','Daniel','Elena','Elenora','Elisa','Federica','Francesca','Gianluca1','Giada','Giorgia',
                'Jonluca','Laura','Leonardo','Linda','Lucrezia','Manu','Marco','Martina','Pagani','Pasquale','Sara',
                'Silvia','Silvia2','Tommaso']


Tmin = 0
Tmax = 3.51
trial_len = 2

rawData = {}
scaler = mne.decoding.Scaler

for s in subject_name:
    save_path = data_path + s+'-coh-epo-'+str(Tmin)+'-'     +str(Tmax)+'-trialLen-'+str(trial_len)+'.fif'
    epochs = mne.read_epochs(save_path)

    rawData[s]=epochs._data[:,:60,:]# ch 59 is speech envelope
    print('----------------------------------------'+s)




with open('raw_data', 'wb') as f:
         pickle.dump(rawData, f)



#input("Press Enter to continue...")



