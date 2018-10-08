import mne
import matplotlib.pyplot as pp
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression as LREG
from mne.decoding import ReceptiveField as RField
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from mne.decoding import Scaler
from mne.time_frequency import tfr_multitaper as TF




def k_fold(data,k=5):
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(data)

    train_ii=[]
    test_ii=[]
    for train_index, test_index in kf.split(data):
        train_ii.append(train_index)
        test_ii.append(test_index)

    return train_ii, test_ii



def Band_filtering(X_orig, band):
    filt_params = dict(order=4, ftype='butter')
    if band=='original':
        X_filtered=X_orig
        return X_filtered
    elif band == 'delta':
        X_filtered = X_orig.filter(1, 4, method='iir', iir_params=filt_params)
        return X_filtered
    elif band == 'theta':
        X_filtered = X_orig.filter(4, 8, method='iir', iir_params=filt_params)
        return X_filtered
    elif band == 'alfa':
        X_filtered = X_orig.filter(8, 12, method='iir', iir_params=filt_params)
        return X_filtered



def use_FreqBand(X_orig, band):

    if band=='original':
        X_freq=X_orig
        return X_freq

    elif band == 'delta':
        freqs = [1, 1.5, 2, 2.5, 3, 3.5, 4]
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





def decoding(band,regularization,tmin,tmax,n_fold,subject_name, savepath):

    data_path = "./ProcessedData/Final_"
    eeg="_processed-epo.fif"
    features="_Features-epo.fif"

    sfreq=100

    n_delays = int((tmax - tmin) * sfreq) + 1

    T= [51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151]


    results_speech= np.zeros((len(regularization),len(T)))# each raw is the results' vector for one regularization parameter
    results_lips= np.zeros((len(regularization),len(T)))# each raw is the results' vector for one regularization parameter


    results_speech_all_sub={}
    results_lips_all_sub={}
    predictions_lips_all_sub={}
    predictions_speech_all_sub={}


    for s in subject_name:

        print('subject '+str(s))

        X_orig = use_FreqBand(mne.read_epochs(data_path+s+eeg),band)
        Features_orig = use_FreqBand(mne.read_epochs(data_path + s + features),band)
        if band=='original':
            X_orig=X_orig.get_data() # 3d array (N_trial, N_channel, N_time)
            Y_envelope_sp_orig=Features_orig.get_data()[:,0,:] # 2d array (N_trial,  N_time)
            Y_lips_ap_orig=Features_orig.get_data()[:,2,:] # 2d array (N_trial,  N_time)
        else:
            X_orig= np.mean(X_orig.data,2)          # 3d array (N_trial, N_channel, N_time)  #averaging power across frequencies
            Y_envelope_sp_orig=np.mean(Features_orig.data[:,0,:,:],1)
            Y_lips_ap_orig=np.mean(Features_orig.data[:,2,:,:],1)
        time = mne.read_epochs(data_path + s + features).times # 1d array (N_time)
        channels = mne.read_epochs(data_path + s + eeg).ch_names

        predictions_speech = np.zeros((Y_envelope_sp_orig.shape[0], 200, len(T),len(regularization)))
        predictions_lips = np.zeros((Y_lips_ap_orig.shape[0],200,len(T),len(regularization)))

        train_index, test_index = k_fold(Y_envelope_sp_orig,n_fold) # define index for train and test for each of the k folds

        #data standardizers
        eegScaler= Scaler(scalings='mean')
        speechScaler= Scaler(scalings='mean')
        lipsScaler = Scaler(scalings='mean')

        scores_speech = np.zeros((n_fold,))
        scores_lips = np.zeros((n_fold,))

        coefs_speech = np.zeros((n_fold, X_orig.shape[1], n_delays))
        patterns_speech = coefs_speech.copy()
        coefs_lips = np.zeros((n_fold, X_orig.shape[1], n_delays))
        patterns_lips = coefs_lips.copy()



        for i, r in enumerate(regularization):

           rf_speech = RField(tmin, tmax, sfreq, feature_names=channels, scoring='r2', patterns=True, estimator=r)
           rf_lips = RField(tmin, tmax, sfreq, feature_names=channels, scoring='r2', patterns=True, estimator=r)

           print('reg parameter #'+str(i))

           for j, t_start in enumerate(T): ##estracting the temporal interval of interest

                t_end= t_start+200
                X = X_orig[:,:,t_start:t_end] #only the eeg window is shifting
                Y_envelope_sp = Y_envelope_sp_orig[:,101:301]
                Y_lips_ap = Y_lips_ap_orig[:,101:301]



                for k in range(0,n_fold):

                    #####COPY X AND Y VARIABLES

                    X_standard=np.zeros((X.shape))
                    Y_lips_ap_standard=np.zeros((Y_lips_ap.shape))
                    Y_envelope_sp_standard = np.zeros((Y_envelope_sp.shape))

                    #standardazing data
                    X_standard[train_index[k], :, :] = eegScaler.fit_transform(X[train_index[k], :, :])
                    X_standard[test_index[k], :, :] = eegScaler.transform(X[test_index[k], :, :])
                    Y_lips_ap_standard[train_index[k], :] = lipsScaler.fit_transform(Y_lips_ap[train_index[k], :])[:,:,0]
                    Y_lips_ap_standard[test_index[k], :] = lipsScaler.transform(Y_lips_ap[test_index[k], :])[:,:,0]
                    Y_envelope_sp_standard[train_index[k], :] = speechScaler.fit_transform(Y_envelope_sp[train_index[k], :])[:,:,0]
                    Y_envelope_sp_standard[test_index[k], :] = speechScaler.transform(Y_envelope_sp[test_index[k], :])[:,:,0]

                    #shaping data as desired by the decoding model (receptive field function)
                    X_standard = np.rollaxis(X_standard, 2, 0)
                    Y_envelope_sp_standard = np.rollaxis(Y_envelope_sp_standard, 1, 0)
                    Y_lips_ap_standard = np.rollaxis(Y_lips_ap_standard, 1, 0)


                    X_TRAIN= X_standard[:,train_index[k],:]
                    X_TEST= X_standard[:,test_index[k],:]
                    Y_envelope_sp_TRAIN = Y_envelope_sp_standard[:,train_index[k]]
                    Y_envelope_sp_TEST = Y_envelope_sp_standard[:,test_index[k]]
                    Y_lips_ap_TRAIN = Y_lips_ap_standard[:,train_index[k]]
                    Y_lips_ap_TEST = Y_lips_ap_standard[:,test_index[k]]

                    #training models and predict
                    rf_speech.fit(X_TRAIN,Y_envelope_sp_TRAIN)
                    rf_lips.fit(X_TRAIN,Y_lips_ap_TRAIN)

                    reconstructed_speech = rf_speech.predict(X_TEST)
                    reconstructed_lips = rf_lips.predict(X_TEST)

                    predictions_speech[test_index[k],:,j,i]=reconstructed_speech.T
                    predictions_lips[test_index[k],:,j,i]=reconstructed_lips.T


                    #computing scores
                    tmp_score_speech=0
                    tmp_score_lips = 0

                    for n, rec in enumerate(reconstructed_speech[:,:,0].T):
                        tmp_score_speech = tmp_score_speech + mean_squared_error(Y_envelope_sp_TEST[:,n]/max(abs(Y_envelope_sp_TEST[:,n])), rec/max(abs(rec)))
                    scores_speech[k]= tmp_score_speech/(n+1)

                    for n, rec in enumerate(reconstructed_lips[:,:,0].T):
                        tmp_score_lips = tmp_score_lips + mean_squared_error(Y_lips_ap_TEST[:, n]/max(abs(Y_lips_ap_TEST[:, n])), rec/max(abs(rec)))
                    scores_lips[k] = tmp_score_lips / (n+1)

                    # scores_speech[k] = rf_speech.score(X_TEST,Y_envelope_sp_TEST)[0]
                    # scores_lips[k] = rf_speech.score(X_TEST,Y_lips_ap_TEST)[0]


                    ##coef_ is shape (n_outputs, n_features, n_delays).
                    # coefs_speech[k] = rf_speech.coef_[0, :, :]
                    # patterns_speech[k] = rf_speech.patterns_[0, :, :]

                    # coefs_lips[k] = rf_lips.coef_[0, :, :]
                    # patterns_lips[k] = rf_lips.patterns_[0, :, :]

                # mean_coefs_lips = coefs_lips.mean(axis=0)
                # mean_patterns_lips = patterns_lips.mean(axis=0)

                mean_scores_lips = scores_lips.mean(axis=0)


                # mean_coefs_speech = coefs_speech.mean(axis=0)
                # mean_patterns_speech = patterns_speech.mean(axis=0)

                mean_scores_speech = scores_speech.mean(axis=0)

                #saving results for the i-th reg parameter and j-th time lag
                results_speech[i, j] = mean_scores_speech
                results_lips[i, j] = mean_scores_lips


        results_speech_all_sub[s]=results_speech.copy()
        results_lips_all_sub[s]=results_lips.copy()
        predictions_speech_all_sub[s]=predictions_speech.copy()
        predictions_lips_all_sub[s]=predictions_lips.copy()




    np.save(savepath+'/results_speech_all_sub',results_speech_all_sub)
    np.save(savepath+'/results_lips_all_sub',results_lips_all_sub)
    np.save(savepath+'/predictions_speech_all_sub',predictions_speech_all_sub)
    np.save(savepath+'/predictions_lips_all_sub',predictions_lips_all_sub)



    tmp_results_speech = []
    tmp_results_lips = []
    for N, s in enumerate(subject_name):
        if N ==0:
            tmp_results_speech= np.asarray(results_speech_all_sub[s])
            tmp_results_lips= np.asarray(results_lips_all_sub[s])
        tmp_results_speech=np.dstack((tmp_results_speech, np.asarray(results_speech_all_sub[s])))
        tmp_results_lips=np.dstack((tmp_results_lips,np.asarray(results_lips_all_sub[s])))

    # computing grand average and standard deviation for each time lag
    GAVG_sp = np.reshape(np.mean(tmp_results_speech,2),(len(regularization),11))
    GAVG_lip = np.reshape(np.mean(tmp_results_lips,2),(len(regularization),11))
    GAVG_sp_std = np.reshape(np.std(tmp_results_speech,2),(len(regularization),11))
    GAVG_lip_std = np.reshape(np.std(tmp_results_lips,2),(len(regularization),11))

    np.save(savepath+'/GAVG_sp',GAVG_sp)
    np.save(savepath+'/GAVG_lip',GAVG_lip)
    np.save(savepath+'/GAVG_sp_std',GAVG_sp_std)
    np.save(savepath+'/GAVG_lip_std',GAVG_lip_std)



    ####PLOTTING RESULTS#####
    T = np.reshape(T, (1, len(T)))
    pp.figure(0)
    for n, r in enumerate(regularization):
        pp.errorbar((T[0,:] - 100) * 10, GAVG_sp[n,:], yerr=GAVG_sp_std[n,:])
    pp.legend(regularization)
    pp.title('speech MSE')
    sfig=savepath+'/GAVG_specch.png'
    pp.savefig(fname=sfig)

    pp.figure(1)
    for n, r in enumerate(regularization):
        pp.errorbar((T[0, :] - 100) * 10, GAVG_lip[n, :], yerr=GAVG_lip_std[n, :])
    pp.legend(regularization)
    pp.title('lips MSE')
    sfig = savepath +'/GAVG_lips.png'
    pp.savefig(fname=sfig)


    #pp.show()

    print('bla')




if __name__ == '__main__':
    r=[1e4]
    #r = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    sub = ['Alice', 'Andrea', 'Daniel', 'Elena', 'Elenora', 'Elisa', 'Federica', 'Francesca', 'Gianluca1',
                    'Giada', 'Giorgia',
                    'Jonluca', 'Laura', 'Leonardo', 'Linda', 'Lucrezia', 'Manu', 'Marco', 'Martina', 'Pagani',
                    'Pasquale', 'Sara',
                    'Silvia', 'Silvia2', 'Tommaso']
    # sub = ['Gianluca1']

    s='./results/'
    s1=s+'alfa'
    s2=s+'delta'
    s3=s+'theta'
    # decoding(band='alfa',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s1)
    # decoding(band='delta',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s2)
    # decoding(band='theta',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s3)
    # r = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    s4=s+'original'
    decoding(band='original', regularization=r, tmin=0, tmax=0, n_fold=5, subject_name=sub, savepath=s4)













