import  numpy as np
import  matplotlib.pyplot as pp
from mne.time_frequency import tfr_multitaper as TF
import mne


def use_FreqBand(X_orig, band):

    if band=='original':
        X_freq=X_orig
        return X_freq

    elif band == 'delta':
        freqs = [1, 1.5, 2, 2.5, 3, 3.5, 4,5]
        X_freq=TF(X_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)
        return X_freq

    elif band == 'theta':
        freqs = [4.5,5, 5.5, 6, 6.5, 7, 7.5, 8]
        X_freq=TF(X_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)
        return X_freq

    elif band == 'alfa':
        freqs = [8.5,9, 9.5, 10, 10.5, 11, 11.5, 12]
        X_freq=TF(X_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)
        return X_freq

T= [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

subject_name = ['Alice','Andrea','Daniel','Elena','Elenora','Elisa','Federica','Francesca','Gianluca1','Giada','Giorgia',
                'Jonluca','Laura','Leonardo','Linda','Lucrezia','Manu','Marco','Martina','Pagani','Pasquale','Sara',
                'Silvia','Silvia2','Tommaso']

#
# lips=np.load('results_lips_all_sub.npy').item()
# speech=np.load('results_speech_all_sub.npy').item()
#
# # pp.plot((np.asarray(T)-100)*10,np.reshape(speech,(11,)))
# # pp.plot((np.asarray(T)-100)*10,np.reshape(lips,(11,)))
# # pp.show()
#
# tmp_results_speech=[]
# tmp_results_lips=[]
#
# for N, s in enumerate(subject_name):
#     tmp_results_speech.append(speech[s])
#     tmp_results_lips.append(lips[s])

#     T = np.reshape(T, (1, len(T)))



original_speechGAVG = np.load('results/original/GAVG_sp.npy')
original_lipGAVG = np.load('results/original/GAVG_lip.npy')
delta_speechGAVG = np.load('results/delta/GAVG_sp.npy')
delta_lipGAVG = np.load('results/delta/GAVG_lip.npy')
theta_speechGAVG = np.load('results/theta/GAVG_sp.npy')
theta_lipGAVG = np.load('results/theta/GAVG_lip.npy')
alfa_speechGAVG = np.load('results/alfa/GAVG_sp.npy')
alfa_lipGAVG = np.load('results/alfa/GAVG_lip.npy')

T = np.reshape(T, (1, len(T)))

# pp.figure(0)
# pp.plot((T[0, :] - 100) * 10, original_speechGAVG[0,:])
# pp.title('raw speech')
# pp.figure(1)
# pp.plot((T[0, :] - 100) * 10, original_lipGAVG[0,:])
# pp.title('raw lips')
# pp.figure(2)
# pp.plot((T[0, :] - 100) * 10, delta_speechGAVG[0,:])
# pp.title('delta speech')
# pp.figure(3)
# pp.plot((T[0, :] - 100) * 10, delta_lipGAVG[0,:])
# pp.title('delta lips')
# pp.figure(4)
# pp.plot((T[0, :] - 100) * 10, theta_speechGAVG[0,:])
# pp.title('theta speech')
# pp.figure(5)
# pp.plot((T[0, :] - 100) * 10, theta_lipGAVG[0,:])
# pp.title('theta lips')
# pp.figure(6)
# pp.plot((T[0, :] - 100) * 10, alfa_speechGAVG[0,:])
# pp.title('alfa speech')
# pp.figure(7)
# pp.plot((T[0, :] - 100) * 10, alfa_lipGAVG[0,:])
# pp.title('alfa lips')
#
# pp.show()
#
#

data_path = "./ProcessedData/Final_"
eeg="_processed-epo.fif"
features="_Features-epo.fif"

speech_pred=np.load('results/delta/predictions_speech_all_sub.npy').item()
speech_pred=speech_pred['Linda']


lips_pred=np.load('results/delta/predictions_lips_all_sub.npy').item()
lips_pred=lips_pred['Linda']

speech = np.mean(use_FreqBand(mne.read_epochs(data_path + 'Linda' + features), 'delta').data[:,0,:,:],1)
lips = np.mean(use_FreqBand(mne.read_epochs(data_path + 'Linda' + features), 'delta').data[:,2,:,:],1)


n_trial=90
lag=6 #5 is delay=0
pp.figure(0)
pp.plot(speech_pred[n_trial,:,lag]/max(speech_pred[n_trial,:,lag]),'r') #5 is T=100 (i.e. delay=0),
pp.plot(speech[n_trial,T[0,lag]:T[0,lag]+200]/max(speech[n_trial,T[0,lag]:T[0,5]+200]),'b')
pp.title('speech, red is the predicted one')
pp.show()


pp.figure(1)
pp.plot(lips_pred[n_trial,:,5]/max(lips_pred[n_trial,:,lag]),'r') #5 is T=100 (i.e. delay=0),
pp.plot(lips[n_trial,T[0,5]:T[0,5]+200]/max(lips[n_trial,T[0,lag]:T[0,lag]+200]),'b')
pp.title('lips, red is the predicted one')
pp.show()