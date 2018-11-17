import mne
import matplotlib.pyplot as pp
import numpy as np
from sklearn.model_selection import KFold
from mne.decoding import ReceptiveField as RField
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr as corrcoeff
from scipy.stats import linregress
from mne.decoding import Scaler
from mne.time_frequency import tfr_multitaper as TF
from sklearn.linear_model import LinearRegression as LReg
from sklearn.preprocessing import  StandardScaler as Scaler


def k_fold(data,k=5):
    kf = KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(data)

    train_ii=[]
    test_ii=[]
    for train_index, test_index in kf.split(data):
        train_ii.append(train_index)
        test_ii.append(test_index)

    return train_ii, test_ii

def evaluate(y_true, y_predicted, metric):
    if metric=='R2':
         score=r2_score(y_true, y_predicted, multioutput='uniform_average')
    elif metric=='MSE':
        score = mean_squared_error(y_true, y_predicted, multioutput='uniform_average')
    elif metric=='corrcoeff':
        score = [linregress(y_true[:, i], y_predicted[:, i])[2] for i in range(0, y_true.shape[1])]
        score=np.mean(score)

    return score


def useFreqBand(X_orig, Features_orig, band, method):
    if method=='band_pass':
        [X_freq, Y_envelope_sp_freq,Y_lips_ap_freq] = Band_filtering(X_orig, Features_orig,band)
    elif method=='multitaper':
        [X_freq, Y_envelope_sp_freq,Y_lips_ap_freq] = Multitaper(X_orig, Features_orig, band)

    return np.float16(X_freq), np.float16(Y_envelope_sp_freq),np.float16(Y_lips_ap_freq)


def Band_filtering(X_orig, Features_orig, band):
    filt_params = dict(order=6, ftype='butter')

    if band=='original':
        X_freq=X_orig.get_data()
        Y_envelope_sp_freq = Features_orig.get_data()[:, 0, :]
        Y_lips_ap_freq = Features_orig.get_data()[:, 2, :]

    elif band == 'delta':
        X_freq = X_orig.filter(1, 4, method='iir', iir_params=filt_params)
        Y_envelope_sp_freq = Features_orig.filter(1, 4, method='iir', iir_params=filt_params).get_data()[:, 0, :]
        Y_lips_ap_freq = Features_orig.filter(1, 4, method='iir', iir_params=filt_params).get_data()[:, 2, :]

    elif band == 'theta':
        X_freq = X_orig.filter(4, 8, method='iir', iir_params=filt_params)
        Y_envelope_sp_freq = Features_orig.filter(4, 8, method='iir', iir_params=filt_params).get_data()[:, 0, :]
        Y_lips_ap_freq = Features_orig.filter(4, 8, method='iir', iir_params=filt_params).get_data()[:, 2, :]

    elif band == 'alfa':
        X_freq = X_orig.filter(8, 12, method='iir', iir_params=filt_params)
        Y_envelope_sp_freq = Features_orig.filter(8, 12, method='iir', iir_params=filt_params).get_data()[:, 0, :]
        Y_lips_ap_freq = Features_orig.filter(8, 12, method='iir', iir_params=filt_params).get_data()[:, 2, :]

    return X_freq, Y_envelope_sp_freq,Y_lips_ap_freq



def Multitaper(X_orig, Features_orig, band):

    if band=='original':
        X_freq=X_orig.get_data() # 3d array (N_trial, N_channel, N_time)
        Y_envelope_sp_freq = Features_orig.get_data()[:, 0, :]  # 2d array (N_trial,  N_time)
        Y_lips_ap_freq = Features_orig.get_data()[:, 2, :]  # 2d array (N_trial,  N_time)
    elif band == 'delta':
        freqs = [1, 1.5, 2, 2.5, 3, 3.5, 4]
        X_freq= np.mean(TF(X_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)._data,2)
        Features_freq= TF(Features_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)
        Y_envelope_sp_freq = np.mean(Features_freq._data[:, 0, :, :], 1)
        Y_lips_ap_freq = np.mean(Features_freq._data[:, 2, :, :], 1)
    elif band == 'theta':
        freqs = [4.5,5, 5.5, 6, 6.5, 7, 7.5, 8]
        X_freq= np.mean(TF(X_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)._data,2)
        Features_freq = TF(Features_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)
        Y_envelope_sp_freq = np.mean(Features_freq._data[:, 0, :, :], 1)
        Y_lips_ap_freq = np.mean(Features_freq._data[:, 2, :, :], 1)
    elif band == 'alfa':
        freqs = [8.5,9, 9.5, 10, 10.5, 11, 11.5, 12]
        X_freq= np.mean(TF(X_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)._data,2)
        Features_freq = TF(Features_orig, freqs, 3, time_bandwidth=4.0, use_fft=True, return_itc=False, decim=1, average=False)
        Y_envelope_sp_freq = np.mean(Features_freq._data[:, 0, :, :], 1)
        Y_lips_ap_freq = np.mean(Features_freq._data[:, 2, :, :], 1)

    return X_freq, Y_envelope_sp_freq,Y_lips_ap_freq


def decoding_withKfold(X,Y_speech,Y_lips,n_fold,train_index,test_index,polynomialReg):

    predictions_speech= np.zeros((Y_speech.shape))
    speech = np.zeros((Y_speech.shape))
    predictions_lips=  np.zeros((Y_lips.shape))
    lips = np.zeros((Y_lips.shape))


    scores_speech=np.zeros((n_fold,))

    for k in range(0, n_fold):

        eegScaler = Scaler()
        speechScaler = Scaler()
        lipsScaler = Scaler()

        speechModel = LReg()
        lipsModel = LReg()

        #####COPY X AND Y VARIABLES

        X_standard = np.zeros((X.shape))
        Y_lips_standard = np.zeros((Y_lips.shape))
        Y_speech_standard = np.zeros((Y_speech.shape))

        # standardazing data
        X_standard[train_index[k], :] = eegScaler.fit_transform(X[train_index[k], :])
        X_standard[test_index[k], :] = eegScaler.transform(X[test_index[k], :])

        Y_lips_standard[train_index[k], :] = lipsScaler.fit_transform(Y_lips[train_index[k], :])
        Y_lips_standard[test_index[k], :] = lipsScaler.transform(Y_lips[test_index[k], :])

        Y_speech_standard[train_index[k], :] = speechScaler.fit_transform(Y_speech[train_index[k], :])
        Y_speech_standard[test_index[k], :] = speechScaler.transform(Y_speech[test_index[k], :])

        X_TRAIN = X_standard[ train_index[k], :]
        X_TEST = X_standard[ test_index[k], :]

        Y_envelope_sp_TRAIN = Y_speech_standard[train_index[k], :]
        Y_envelope_sp_TEST = Y_speech_standard[test_index[k], :]

        Y_lips_ap_TRAIN = Y_lips_standard[train_index[k], :]
        Y_lips_ap_TEST = Y_lips_standard[test_index[k], :]


        if polynomialReg == True:
            X_TRAIN= np.concatenate((X_TRAIN,np.power(X_TRAIN,2)),1)
            X_TEST = np.concatenate((X_TEST, np.power(X_TEST, 2)), 1)

        # training models and predict
        speechModel.fit(X_TRAIN, Y_envelope_sp_TRAIN)
        lipsModel.fit(X_TRAIN, Y_lips_ap_TRAIN)

        reconstructed_speech = speechModel.predict(X_TEST)
        reconstructed_lips = lipsModel.predict(X_TEST)

        predictions_speech[test_index[k], :] = reconstructed_speech
        speech[test_index[k], :] = Y_envelope_sp_TEST

        predictions_lips[test_index[k], :] = reconstructed_lips
        lips[test_index[k], :] = Y_lips_ap_TEST

    # computing scores
    speech_score = evaluate(speech.T, predictions_speech.T, 'corrcoeff')
    lips_score = evaluate(lips.T, predictions_lips.T, 'corrcoeff')

    return speech_score, lips_score, predictions_speech, predictions_lips, speech, lips








def decoding(band,regularization,n_fold,subject_name, savepath,polynomialReg=False):

    data_path = "./ProcessedData/Final_"
    eeg="_processed-epo.fif"
    features="_Features-epo.fif"

    sfreq=100

    T= [51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151]


    score_speech_all_sub={}
    score_lips_all_sub={}
    predictions_lips_all_sub={}
    predictions_speech_all_sub={}
    true_lips_all_sub = {}
    true_speech_all_sub = {}

    for s in subject_name:

        print('subject '+str(s))

        X_orig= mne.read_epochs(data_path + s + eeg)
        Features_orig= mne.read_epochs(data_path + s + features)

        [X_orig,Y_envelope_sp_orig,Y_lips_ap_orig]= useFreqBand(X_orig, Features_orig, band, 'band_pass')

        time = mne.read_epochs(data_path + s + features).times # 1d array (N_time)
        channels = mne.read_epochs(data_path + s + eeg).ch_names

        predictions_speech_oneSub = np.zeros((Y_envelope_sp_orig.shape[0], 200, X_orig.shape[1],len(T),len(regularization)),dtype="float16") # dimendions:(n_trials, trial length, n_channels, n_delays, n_reg_params)
        predictions_lips_oneSub = np.zeros((Y_lips_ap_orig.shape[0],200,X_orig.shape[1],len(T),len(regularization)),dtype="float16")
        true_speech_oneSub = np.zeros((Y_envelope_sp_orig.shape[0], 200, X_orig.shape[1],len(T), len(regularization)),dtype="float16")
        true_lips_oneSub = np.zeros((Y_lips_ap_orig.shape[0], 200, X_orig.shape[1],len(T), len(regularization)),dtype="float16")
        score_speech_oneSub=np.zeros((X_orig.shape[1],len(T),len(regularization)),dtype="float16")  # dimendions:(n_channels, n_delays, n_reg_params)
        score_lips_oneSub = np.zeros((X_orig.shape[1],len(T), len(regularization)),dtype="float16")


        train_index, test_index = k_fold(Y_envelope_sp_orig,n_fold) # define index for train and test for each of the k folds


        for c, ch_name in enumerate(channels):

            X_orig_oneCh=X_orig[:,c,:]

            for i, r in enumerate(regularization):

               print('sub '+s+' - reg parameter #'+str(i))

               for j, t_start in enumerate(T): ##estracting the temporal interval of interest

                    t_end= t_start+200
                    X = X_orig_oneCh[:,t_start:t_end] #only the eeg window is shifting
                    Y_envelope_sp = Y_envelope_sp_orig[:,101:301]
                    Y_lips_ap = Y_lips_ap_orig[:,101:301]

                    [speech_score, lips_score, predictions_speechh, predictions_lipss, speech, lips]=decoding_withKfold(X,Y_envelope_sp,Y_lips_ap,n_fold,train_index,test_index,polynomialReg)
                    predictions_lips_oneSub[:,:,c,j,i]=predictions_lipss
                    predictions_speech_oneSub[:,:,c,j,i]=predictions_speechh
                    true_speech_oneSub[:,:,c, j, i]=lips
                    true_lips_oneSub[:,:,c, j, i]=speech
                    score_speech_oneSub[c,j,i]=speech_score
                    score_lips_oneSub[c,j,i] = lips_score

        score_speech_all_sub[s]=score_speech_oneSub.copy()
        score_lips_all_sub[s]=score_lips_oneSub.copy()
        predictions_speech_all_sub[s]=predictions_speech_oneSub.copy()
        predictions_lips_all_sub[s]=predictions_lips_oneSub.copy()
        true_speech_all_sub[s] = true_speech_oneSub.copy()
        true_lips_all_sub[s] = true_lips_oneSub.copy()

    np.save(savepath+'/score_speech_all_sub',score_speech_all_sub)
    np.save(savepath+'/score_lips_all_sub',score_lips_all_sub)
    np.save(savepath+'/predictions_speech_all_sub',predictions_speech_all_sub)
    np.save(savepath+'/predictions_lips_all_sub',predictions_lips_all_sub)
    np.save(savepath + '/true_speech_all_sub', true_speech_all_sub)
    np.save(savepath + '/true_lips_all_sub', true_lips_all_sub)

    tmp_results_speech = []
    tmp_results_lips = []
    for N, s in enumerate(subject_name):
        if N ==0:
            tmp_results_speech= np.expand_dims(score_speech_all_sub[s],axis=-1)
            tmp_results_lips= np.expand_dims(score_lips_all_sub[s],axis=-1)
        else:
            tmp_results_speech = np.concatenate((tmp_results_speech, np.expand_dims(score_speech_all_sub[s],axis=-1)), -1)
            tmp_results_lips = np.concatenate((tmp_results_lips, np.expand_dims(score_lips_all_sub[s],axis=-1)), -1)

    # computing grand average and standard deviation for each time lag
    if N !=0:
        GAVG_sp = np.mean(tmp_results_speech,-1)
        GAVG_lip = np.mean(tmp_results_lips,-1)
        GAVG_sp_std = np.std(tmp_results_speech,-1)
        GAVG_lip_std = np.std(tmp_results_lips,-1)

        np.save(savepath+'/GAVG_sp',GAVG_sp)
        np.save(savepath+'/GAVG_lip',GAVG_lip)
        np.save(savepath+'/GAVG_sp_std',GAVG_sp_std)
        np.save(savepath+'/GAVG_lip_std',GAVG_lip_std)



    # ####PLOTTING RESULTS#####
    # T = np.reshape(T, (1, len(T)))
    # pp.figure(nplot)
    # for n, r in enumerate(regularization):
    #     pp.errorbar((T[0,:] - 100) * 10, GAVG_sp[n,:], yerr=GAVG_sp_std[n,:])
    # pp.legend(regularization)
    # pp.title('speech MSE')
    # sfig=savepath+'/GAVG_specch.png'
    # pp.savefig(fname=sfig)
    #
    # pp.figure(nplot+1)
    # for n, r in enumerate(regularization):
    #     pp.errorbar((T[0, :] - 100) * 10, GAVG_lip[n, :], yerr=GAVG_lip_std[n, :])
    # pp.legend(regularization)
    # pp.title('lips MSE')
    # sfig = savepath +'/GAVG_lips.png'
    # pp.savefig(fname=sfig)
    #
    #
    # pp.show()
    #
    # print('bla')




if __name__ == '__main__':
    r=[1e4]
    #r = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    # sub = ['Alice', 'Andrea', 'Daniel', 'Elena', 'Elenora', 'Elisa', 'Federica', 'Francesca', 'Gianluca1',
    #        'Giada', 'Giorgia', 'Jonluca', 'Laura', 'Leonardo', 'Linda', 'Lucrezia', 'Manu', 'Marco', 'Martina', 'Pagani',
    #        'Pasquale', 'Sara', 'Silvia', 'Silvia2', 'Tommaso']

    sub = ['Gianluca1', 'Sara']

    s='./results_NoTRF/'
    s1=s+'alfa'
    s2=s+'delta'
    s3=s+'theta'
    n1=1
    n2=3
    n3=5
    n4=7
    # decoding(band='alfa',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s1,nplot=n1)
    # decoding(band='delta',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s2,nplot=n2)
    # decoding(band='theta',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s3,nplot=n3)
    # r = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    s4=s+'original'
    decoding(band='original', regularization=r, n_fold=5, subject_name=sub, savepath=s4,polynomialReg=True)
