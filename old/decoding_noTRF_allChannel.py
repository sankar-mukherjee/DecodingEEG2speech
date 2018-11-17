import mne
import matplotlib.pyplot as pp
import numpy as np
from sklearn.model_selection import KFold
from mne.decoding import ReceptiveField as RField
from mne.decoding import Scaler as MultiChannelScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr as corrcoeff
from scipy.stats import linregress
from mne.decoding import Scaler
from mne.time_frequency import tfr_multitaper as TF
from sklearn.linear_model import LinearRegression as LReg
from sklearn.preprocessing import  StandardScaler as Scaler
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.metrics import explained_variance_score

def pca_decomposition(data,keepVar=0.95):
    pca = PCA(n_components=data.shape[1])
    varSum = 0
    n_comp = 0
    pca.fit(data)
    for value in pca.explained_variance_ratio_:
        varSum = varSum + value
        n_comp = n_comp + 1
        if varSum >= keepVar:
            break
    return pca, n_comp

def kernel_pca_decomposition(data,keepVar=0.95,kernel='rbf'):
    pca = KernelPCA(kernel=kernel,n_components=data.shape[1],n_jobs=-1)
    varSum = 0
    n_comp = 0
    pca.fit(data)
    for value in pca.explained_variance_ratio_:
        varSum = varSum + value
        n_comp = n_comp + 1
        if varSum >= keepVar:
            break
    return pca, n_comp

def ordered_ICAcomps(data,MixingMat,demixedData):
    var_scores=np.zeros((demixedData.shape[1],2))
    for i in range(0,demixedData.shape[1]):
        var_scores[i,0]=explained_variance_score(data, np.matmul(np.expand_dims(demixedData[:, i], 1), np.expand_dims(MixingMat[:, i], 1).T))
        var_scores[i,1]=i
    var_scores = var_scores[np.flip(var_scores[:, 0].argsort(),0)]
    return var_scores



def ICA_decomposition(data,keepVar=0.95):
    ica=FastICA(whiten = True, max_iter = 300, tol = 0.0001)
    demixed=ica.fit_transform(data)
    M=ica.mixing_
    var_scores=ordered_ICAcomps(data, M, demixed)
    varSum=0
    n_comp=0
    for value in var_scores[:,0]:
        varSum = varSum + value
        n_comp = n_comp + 1
        if varSum >= keepVar:
            break
    selected_comps=var_scores[:n_comp,1]
    return ica, selected_comps



    remix=ica.inverse_transform(demixed)
    varSum = 0
    for nc in range(0,data.shape[1]):
        varSum = varSum + explained_variance_score(data,data-demixed[:,nc])
        n_comp = n_comp + 1
        if varSum >= keepVar:
            break
    return ica, n_comp


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

    return np.float32(X_freq), np.float32(Y_envelope_sp_freq),np.float32(Y_lips_ap_freq)


def Band_filtering(X_orig, Features_orig, band):
    X_orig._data = X_orig.get_data() * 1e6
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
    X_orig._data = X_orig.get_data() * 1e6
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


def decoding_withKfold(X,Y_speech,Y_lips,n_fold,train_index,test_index,examples,feature):

    predictions_speech= np.zeros((Y_speech.shape))
    speech = np.zeros((Y_speech.shape))
    predictions_lips=  np.zeros((Y_lips.shape))
    lips = np.zeros((Y_lips.shape))


    scores_speech=np.zeros((n_fold,))

    for k in range(0, n_fold):

        eegScaler = MultiChannelScaler(scalings='mean')
        speechScaler = MultiChannelScaler(scalings='mean')
        lipsScaler = MultiChannelScaler(scalings='mean')

        speechModel = LReg()
        lipsModel = LReg()

        #####COPY X AND Y VARIABLES

        X_standard = np.zeros((X.shape))
        Y_lips_standard = np.zeros((Y_lips.shape))
        Y_speech_standard = np.zeros((Y_speech.shape))

        # standardazing data
        X_standard[train_index[k], :,:] = eegScaler.fit_transform(X[train_index[k], :, :])
        X_standard[test_index[k], :,:] = eegScaler.transform(X[test_index[k], :, :])

        Y_lips_standard[train_index[k], :] = lipsScaler.fit_transform(Y_lips[train_index[k], :]).squeeze()
        Y_lips_standard[test_index[k], :] = lipsScaler.transform(Y_lips[test_index[k], :]).squeeze()

        Y_speech_standard[train_index[k], :] = speechScaler.fit_transform(Y_speech[train_index[k], :]).squeeze()
        Y_speech_standard[test_index[k], :] = speechScaler.transform(Y_speech[test_index[k], :]).squeeze()

        X_TRAIN = X_standard[ train_index[k], :,:]
        X_TEST = X_standard[ test_index[k], :,:]

        Y_envelope_sp_TRAIN = Y_speech_standard[train_index[k], :]
        Y_envelope_sp_TEST = Y_speech_standard[test_index[k], :]

        Y_lips_ap_TRAIN = Y_lips_standard[train_index[k], :]
        Y_lips_ap_TEST = Y_lips_standard[test_index[k], :]

        #X_train and test now are (#trials,#channnels,#timepoints)
        n_trial=X_TRAIN.shape[0]
        n_trial_test=X_TEST.shape[0]
        n_ch = X_TRAIN.shape[1]
        trial_length = X_TRAIN.shape[2]

        if examples=='are_Trials':
            X_TRAIN_tmp=np.zeros((X_TRAIN.shape[0],n_ch*trial_length))
            X_TEST_tmp=np.zeros((X_TEST.shape[0],n_ch*trial_length))
            for i in range(0,n_ch):
                X_TRAIN_tmp[:,i*trial_length:(i+1)*trial_length]=X_TRAIN[:,i,:]
                X_TEST_tmp[:, i * trial_length:(i + 1) * trial_length] = X_TEST[:, i, :]
            X_TRAIN=X_TRAIN_tmp
            X_TEST=X_TEST_tmp

        elif examples=='are_Time':
            X_TRAIN_tmp = np.zeros((n_trial*trial_length, n_ch))
            X_TEST_tmp = np.zeros((n_trial_test*trial_length, n_ch))
            Y_envelope_sp_TRAIN_tmp = np.zeros((n_trial * trial_length, ))
            Y_envelope_sp_TEST_tmp = np.zeros((n_trial_test * trial_length, ))
            Y_lips_ap_TRAIN_tmp = np.zeros((n_trial * trial_length, ))
            Y_lips_ap_TEST_tmp = np.zeros((n_trial_test * trial_length, ))
            for i in range(0,n_trial):
                X_TRAIN_tmp[i*trial_length:(i+1)*trial_length,:]= X_TRAIN[i, :, :].T
                Y_envelope_sp_TRAIN_tmp[i*trial_length:(i+1)*trial_length]=Y_envelope_sp_TRAIN[i,:]
                Y_lips_ap_TRAIN_tmp[i * trial_length:(i + 1) * trial_length] = Y_lips_ap_TRAIN[i, :]
                if i<X_TEST.shape[0]: #test trials are less than train
                    X_TEST_tmp[i*trial_length:(i+1)*trial_length,:] = X_TEST[i, :, :].T
                    Y_envelope_sp_TEST_tmp[i * trial_length:(i + 1) * trial_length] = Y_envelope_sp_TEST[i, :]
                    Y_lips_ap_TEST_tmp[i * trial_length:(i + 1) * trial_length] = Y_lips_ap_TEST[i, :]
            X_TRAIN=X_TRAIN_tmp
            X_TEST=X_TEST_tmp
            Y_envelope_sp_TRAIN=Y_envelope_sp_TRAIN_tmp
            Y_envelope_sp_TEST=Y_envelope_sp_TEST_tmp
            Y_lips_ap_TRAIN=Y_lips_ap_TRAIN_tmp
            Y_lips_ap_TEST=Y_lips_ap_TEST_tmp

            if feature == 'pca':
                [pca,n_comp]=pca_decomposition(X_TRAIN)
                X_TRAIN=pca.transform(X_TRAIN)[:,:n_comp]
                X_TEST=pca.transform(X_TEST)[:,:n_comp]
            if feature == 'Kpca':
                [pca,n_comp]=kernel_pca_decomposition(X_TRAIN)
                X_TRAIN=pca.transform(X_TRAIN)[:,:n_comp]
                X_TEST=pca.transform(X_TEST)[:,:n_comp]
            if feature =='ica':
                ICA_decomposition
                [ica, selected_comps] = ICA_decomposition(X_TRAIN)
                X_TRAIN=ica.transform(X_TRAIN)[:,selected_comps.astype('int')]
                X_TEST=ica.transform(X_TEST)[:,selected_comps.astype('int')]

            if feature =='derivative1':
                de1=np.diff(X_TRAIN,axis=0)/0.01
                de1= np.concatenate( (np.zeros((1,de1.shape[1])),de1),axis=0)
                for i in range(0, de1.shape[0],trial_length):
                    de1[i,:]= np.zeros((1,de1.shape[1]))
                X_TRAIN = np.concatenate((X_TRAIN, de1), 1)

                de1 = np.diff(X_TEST, axis=0)/0.01
                de1 = np.concatenate((np.zeros((1, de1.shape[1])), de1), axis=0)
                for i in range(0, de1.shape[0], trial_length):
                    de1[i, :] = np.zeros((1, de1.shape[1]))
                X_TEST = np.concatenate((X_TEST, de1), 1)

            if feature == 'derivative2':
                de1 = np.diff(X_TRAIN, axis=0)/0.01
                de1 = np.concatenate((np.zeros((1, de1.shape[1])), de1), axis=0)
                for i in range(0, de1.shape[0], trial_length):
                    de1[i, :] = np.zeros((1, de1.shape[1]))

                de2 = np.diff(de1, axis=0)
                de2 = np.concatenate((np.zeros((1, de2.shape[1])), de2), axis=0)
                for i in range(0, de2.shape[0], trial_length):
                    de2[i, :] = np.zeros((1, de2.shape[1]))
                    de2[i+1, :] = np.zeros((1, de2.shape[1]))

                X_TRAIN = np.concatenate( (np.concatenate((X_TRAIN, de1), 1),de2),1)

                de1 = np.diff(X_TEST, axis=0)/0.01
                de1 = np.concatenate((np.zeros((1, de1.shape[1])), de1), axis=0)
                for i in range(0, de1.shape[0], trial_length):
                    de1[i, :] = np.zeros((1, de1.shape[1]))

                de2 = np.diff(de1, axis=0)
                de2 = np.concatenate((np.zeros((1, de2.shape[1])), de2), axis=0)
                for i in range(0, de2.shape[0], trial_length):
                    de2[i, :] = np.zeros((1, de2.shape[1]))
                    de2[i+1, :] = np.zeros((1, de2.shape[1]))

                X_TEST = np.concatenate( (np.concatenate((X_TEST, de1), 1),de2),1)


        if feature == 'polynomial':
                X_TRAIN = np.concatenate((X_TRAIN, np.power(X_TRAIN, 2)), 1)
                X_TEST = np.concatenate((X_TEST, np.power(X_TEST, 2)), 1)



        # training models and predict
        speechModel.fit(X_TRAIN, Y_envelope_sp_TRAIN)
        lipsModel.fit(X_TRAIN, Y_lips_ap_TRAIN)

        reconstructed_speech = speechModel.predict(X_TEST)
        reconstructed_lips = lipsModel.predict(X_TEST)

        if examples=='are_Time':
            reconstructed_speech_tmp=np.zeros((n_trial_test,trial_length))
            reconstructed_lips_tmp=np.zeros((n_trial_test,trial_length))
            Y_envelope_sp_TEST_tmp=np.zeros((n_trial_test,trial_length))
            Y_lips_ap_TEST_tmp=np.zeros((n_trial_test,trial_length))
            t=0
            for i in range(0,len(reconstructed_speech),trial_length):
                reconstructed_speech_tmp[t,:]=reconstructed_speech[i:i+trial_length]
                reconstructed_lips_tmp[t,:]=reconstructed_lips[i:i+trial_length]
                Y_envelope_sp_TEST_tmp[t,:]=Y_envelope_sp_TEST[i:i+trial_length]
                Y_lips_ap_TEST_tmp[t,:]=Y_lips_ap_TEST[i:i+trial_length]
                t+=1
            reconstructed_speech=reconstructed_speech_tmp
            reconstructed_lips=reconstructed_lips_tmp
            Y_envelope_sp_TEST=Y_envelope_sp_TEST_tmp
            Y_lips_ap_TEST=Y_lips_ap_TEST_tmp

        predictions_speech[test_index[k], :] = reconstructed_speech
        speech[test_index[k], :] = Y_envelope_sp_TEST

        predictions_lips[test_index[k], :] = reconstructed_lips
        lips[test_index[k], :] = Y_lips_ap_TEST

    # computing scores
    speech_score = evaluate(speech.T, predictions_speech.T, 'corrcoeff')
    lips_score = evaluate(lips.T, predictions_lips.T, 'corrcoeff')

    return speech_score, lips_score, predictions_speech, predictions_lips, speech, lips








def decoding(band,frequencyExtraction,regularization,n_fold,subject_name, savepath,examples='are_Trials',feature='linear'):

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

        if 'all' in savepath:
            X_orig= mne.read_epochs(data_path + s + eeg)
        elif 'best' in savepath:
            pick=['FC3','FC5','C3','C5','C4','C6','FC4','FC6','F5','F3','F4','F6']
            #pick = ['FC6']
            X_orig = mne.read_epochs(data_path + s + eeg).pick_channels(pick)
        Features_orig= mne.read_epochs(data_path + s + features)


        [X_orig,Y_envelope_sp_orig,Y_lips_ap_orig]= useFreqBand(X_orig, Features_orig, band, frequencyExtraction)

        time = mne.read_epochs(data_path + s + features).times # 1d array (N_time)
        channels = mne.read_epochs(data_path + s + eeg).ch_names

        predictions_speech_oneSub = np.zeros((Y_envelope_sp_orig.shape[0], 200, len(T),len(regularization)),dtype="float32") # dimendions:(n_trials, trial length, n_channels, n_delays, n_reg_params)
        predictions_lips_oneSub = np.zeros((Y_lips_ap_orig.shape[0],200,len(T),len(regularization)),dtype="float32")
        true_speech_oneSub = np.zeros((Y_envelope_sp_orig.shape[0], 200, len(T), len(regularization)),dtype="float32")
        true_lips_oneSub = np.zeros((Y_lips_ap_orig.shape[0], 200, len(T), len(regularization)),dtype="float32")
        score_speech_oneSub=np.zeros((len(T),len(regularization)),dtype="float32")  # dimendions:(n_channels, n_delays, n_reg_params)
        score_lips_oneSub = np.zeros((len(T), len(regularization)),dtype="float32")

        train_index, test_index = k_fold(Y_envelope_sp_orig,n_fold) # define index for train and test for each of the k folds

        for i, r in enumerate(regularization):

           print('sub '+s+' - reg parameter #'+str(i))

           for j, t_start in enumerate(T): ##estracting the temporal interval of interest

                t_end= t_start+200
                X = X_orig[:,:,t_start:t_end] #only the eeg window is shifting
                Y_envelope_sp = Y_envelope_sp_orig[:,101:301]
                Y_lips_ap = Y_lips_ap_orig[:,101:301]

                [speech_score, lips_score, predictions_speechh, predictions_lipss, speech, lips]=decoding_withKfold(X,Y_envelope_sp,Y_lips_ap,n_fold,train_index,test_index,examples,feature)
                predictions_lips_oneSub[:,:,j,i]=predictions_lipss
                predictions_speech_oneSub[:,:,j,i]=predictions_speechh
                true_speech_oneSub[:,:, j, i]=lips
                true_lips_oneSub[:,:, j, i]=speech
                score_speech_oneSub[j,i]=speech_score
                score_lips_oneSub[j,i] = lips_score

        score_speech_all_sub[s]=score_speech_oneSub.copy()
        score_lips_all_sub[s]=score_lips_oneSub.copy()
        predictions_speech_all_sub[s]=predictions_speech_oneSub.copy()
        predictions_lips_all_sub[s]=predictions_lips_oneSub.copy()
        true_speech_all_sub[s] = true_speech_oneSub.copy()
        true_lips_all_sub[s] = true_lips_oneSub.copy()

    np.save(savepath+'/score_speech_all_sub_'+examples+'_'+frequencyExtraction+'_'+feature,score_speech_all_sub)
    np.save(savepath+'/score_lips_all_sub_'+examples+'_'+frequencyExtraction+'_'+feature,score_lips_all_sub)
    np.save(savepath+'/predictions_speech_all_sub_'+examples+'_'+frequencyExtraction+'_'+feature,predictions_speech_all_sub)
    np.save(savepath+'/predictions_lips_all_sub_'+examples+'_'+frequencyExtraction+'_'+feature,predictions_lips_all_sub)
    np.save(savepath + '/true_speech_all_sub_'+examples+'_'+frequencyExtraction+'_'+feature, true_speech_all_sub)
    np.save(savepath + '/true_lips_all_sub_'+examples+'_'+frequencyExtraction+'_'+feature, true_lips_all_sub)

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

        np.save(savepath+'/GAVG_sp_'+examples+'_'+frequencyExtraction+'_'+feature,GAVG_sp)
        np.save(savepath+'/GAVG_lip_'+examples+'_'+frequencyExtraction+'_'+feature,GAVG_lip)
        np.save(savepath+'/GAVG_sp_std_'+examples+'_'+frequencyExtraction+'_'+feature,GAVG_sp_std)
        np.save(savepath+'/GAVG_lip_std_'+examples+'_'+frequencyExtraction+'_'+feature,GAVG_lip_std)



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
    #####PARAMETERS####
    #feature='polynomial' or 'linear' and ONLY FOR regmethd='areTime' 'ica' or 'pca' or 'derivative1' or 'derivative2'
    #fmethod = 'multitaper' or 'band_pass'
    #regmethod='are_Trials' or 'are_Time'
    #fband='delta'or'theta'or'alfa'or'original'
    #path i.e. s must be not changed, except for the word best (if you want the best cluster) tha can be also all (using all channels)
    # r is not use since only linear reg is implemented now

    r=[1e4]
    #r = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    sub = ['Alice', 'Andrea', 'Daniel', 'Elena', 'Elenora', 'Elisa', 'Federica', 'Francesca', 'Gianluca1',
           'Giada', 'Giorgia', 'Jonluca', 'Laura', 'Leonardo', 'Linda', 'Lucrezia', 'Manu', 'Marco', 'Martina', 'Pagani',
           'Pasquale', 'Sara', 'Silvia', 'Silvia2', 'Tommaso']


    #sub = ['Gianluca1', 'Sara']

    fband='delta'
    fmethod='multitaper'
    regmethod='are_Time'
    feature='derivative1'

    # decoding(band='alfa',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s1,nplot=n1)
    # decoding(band='delta',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s2,nplot=n2)
    # decoding(band='theta',regularization=r,tmin=0,tmax=0,n_fold=5,subject_name=sub,savepath=s3,nplot=n3)
    # r = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]

    s='./results_NoTRF/best_channels/'
    sbest=s+fband#+'/FC6'

    decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod,feature=feature)
    feature='derivative2'
    decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod,feature=feature)
    fmethod = 'band_pass'
    decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod,feature=feature)
    feature = 'derivative1'
    decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod,feature=feature)


    # regmethod = 'are_Trials'
    # decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod,feature=feature)
    # feature = 'polynomial'
    # decoding(band=fband, frequencyExtraction=fmethod, regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod, feature=feature)
    # fmethod='multitaper'
    # decoding(band=fband, frequencyExtraction=fmethod, regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod, feature=feature)
    # feature = 'linear'
    # decoding(band=fband, frequencyExtraction=fmethod, regularization=r, n_fold=3, subject_name=sub, savepath=sbest,examples=regmethod, feature=feature)

    # s = './results_NoTRF/all_channels/'
    # sall=s+fband
    # decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sall,examples=regmethod,feature=feature)
    # fmethod = 'multitaper'
    # decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sall,examples=regmethod,feature=feature)





    # s='./results_NoTRF/best_channels/'
    #
    # decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sp,examples=regmethod,feature=feature,use_someCh=False)
    # fmethod='multitaper'
    #
    # decoding(band=fband, frequencyExtraction=fmethod,regularization=r, n_fold=3, subject_name=sub, savepath=sp,examples=regmethod,feature=feature,use_someCh=False)
