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
from mne.preprocessing import ICA
import os





warnings.filterwarnings('ignore')

data_path = "./coherence_epochs/"

subject_name = ['Alice','Andrea','Daniel','Elena','Elenora','Elisa','Federica','Francesca','Gianluca1','Giada','Giorgia',
                'Jonluca','Laura','Leonardo','Linda','Lucrezia','Manu','Marco','Martina','Pagani','Pasquale','Sara',
                'Silvia','Silvia2','Tommaso']
# subject_name = ['Alice']


Tmin = 0
Tmax = 3.51
trial_len = 2


remove_ch = ["Fpz", "Iz","F9", "F10", "P9", "P10", "O9", "O10", "AFz","FT9","FT10","TP9","TP10","PO9","PO10"]
montage = mne.channels.read_montage('easycap-M1')
ch_names = montage.ch_names
a = np.nonzero(np.in1d(ch_names, remove_ch))[0]
ch_names = np.delete(ch_names,a)
montage = mne.channels.read_montage('easycap-M1',ch_names)

filt_params = dict(order=5 , ftype='butter')
ica=ICA(method='fastica')

for s in subject_name:
        if not os.path.isfile("ProcessedData/" + s + "_processed-epo.fif") :
            path = data_path + s+'-coh-epo-'+str(Tmin)+'-'     +str(Tmax)+'-trialLen-'+str(trial_len)+'.fif'
            epochs = mne.read_epochs(path)

            notEEG=epochs.copy().drop_channels(montage.ch_names)
            epochs.drop_channels(notEEG.ch_names)
            epochs.set_montage(montage)

            epochs_filtered= epochs.copy().filter( 1, 40,  method='iir', iir_params=filt_params)
            epochs_filtered.plot(n_epochs=10 ) #to remove bad epochs
            # epochs.filter(1., 40., fir_design='firwin')


            #
            # for ii, ep in enumerate(epochs_filtered.iter_evoked()): ######TO VISUALIZE SINGLE EPOCH TOPOGRAPHY
            #     if ii=='X': #CHANGE X WITH THE EPOCH'S NUMBER
            #         ep.plot_topomap(times='interactive' )


            # epochs_filtered_copy = epochs_filtered.copy()
            # ica.fit(epochs_filtered,picks=range(0,59))
            # ok=False
            # while ok != True:
            #     ica.plot_sources(epochs_filtered)
            #     ica.apply(epochs_filtered)
            #
            #     epochs_filtered_copy.plot(picks=range(0,59))
            #     epochs_filtered.plot(picks=range(0,59))
            #     ok=input("y if ok, other keys to try the rejection of ica components again")
            #     if ok != "y":
            #         epochs_filtered=epochs_filtered_copy.copy()
            #     else: break




          #  rawData[s]=epochs._data[:,:60,:]# ch 59 is speech envelope
            epochs_filtered.save("ProcessedData/" + s + "_processed-epo.fif")
            print('----------------------------------------'+s)




for s in subject_name:
    if not os.path.isfile("ProcessedData/" + s + "_Features-epo.fif"):
            path ="ProcessedData/" + s  + "_processed-epo.fif"
            path_raw = data_path + s+'-coh-epo-'+str(Tmin)+'-'     +str(Tmax)+'-trialLen-'+str(trial_len)+'.fif'
            ep = mne.read_epochs(path)
            ep_raw = mne.read_epochs(path_raw)
            notEEG = ep_raw.copy().drop_channels(montage.ch_names)

            tot= range(0,ep_raw.selection[-1]+1)
            pre_eliminate=[]
            for i in tot:
                if i not in ep_raw.selection:
                    pre_eliminate.append(i)

            post_eliminate = []
            for i in ep_raw.selection:
                if i not in ep.selection:
                    post_eliminate.append(i)


            if pre_eliminate:
                print(np.asarray(pre_eliminate)+1)

            if post_eliminate:
                print(np.asarray(post_eliminate)+1)

            # ep_raw.plot()
            notEEG.plot()
            notEEG.save("ProcessedData/" + s + "_Features-epo.fif")
            # ep.plot()
            print('i')


data_path = "./ProcessedData/"

eeg="_processed-epo.fif"
features="_Features-epo.fif"

###MATLAB CONVERSION
# for s in subject_name:
#     X = mne.read_epochs(data_path + s + eeg)
#     X = mne.read_epochs(data_path+s+eeg).crop(tmin=0, tmax=3.51).resample(100).get_data() # 3d array (N_trial, N_channel, N_time)
#     time= mne.read_epochs(data_path+s+eeg).crop(tmin=0, tmax=3.51).resample(100).times
#     Y_envelope_sp = mne.read_epochs(data_path+s+features).crop(tmin=0, tmax=3.51).resample(100).get_data()[:,0,:] # 2d array (N_trial,  N_time)
#     Y_lips_ap = mne.read_epochs(data_path + s + features).crop(tmin=0, tmax=3.51).resample(100).get_data()[:,2,:] # 2d array (N_trial,  N_time)
#     scipy.io.savemat(s+'_TrialsProcessed', dict(EEG=X, SPEECH_ENV=Y_envelope_sp, LIPS_AP=Y_lips_ap,TIME=time))


for s in subject_name:
    X = mne.read_epochs(data_path+s+eeg).crop(tmin=0, tmax=3.51).resample(100).save(data_path + "Final_" + s + eeg)
    Y_envelope_sp = mne.read_epochs(data_path+s+features).crop(tmin=0, tmax=3.51).resample(100).save(data_path + "Final_" + s + features)
