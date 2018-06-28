import numpy as np
import pickle
import tensorflow as tf
import sklearn as sk
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import mne
from matplotlib import pyplot as plt



with open('raw_data', 'rb') as f:
     Data = pickle.load(f)

#sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))

# subject_name = ['Alice','Andrea','Daniel','Elena','Elenora','Elisa','Federica','Francesca','Gianluca1','Giada','Giorgia',
#                 'Jonluca','Laura','Leonardo','Linda','Lucrezia','Manu','Marco','Martina','Pagani','Pasquale','Sara',
#                 'Silvia','Silvia2','Tommaso']
subject_name = ['Alice']

epochsNormalized={}

train_eeg={}
train_examples={}

train_speech={}
train_targets={}

test_eeg={}
test_examples={}
test_speech={}
test_targets={}

for s in subject_name:
    #scaler = mne.decoding.Scaler(scalings='mean').fit(epochs_data=Data[s])
    #tmp = scaler.transform(Data[s])
    epochsNormalized[s] = Data[s]
    print('----------------------------------------' + s)
    train_eeg[s] = epochsNormalized[s][:100, :-1, :]   #### [ trials, channels, time]
    test_eeg[s] = epochsNormalized[s][100:, :-1, :]

    train_speech[s] = epochsNormalized[s][:100,-1,:] #speech
    test_speech[s] = epochsNormalized[s][100:,-1,:]

    ntrials, nchannels, ntime = train_eeg[s].shape
    train_examples[s] = np.zeros((ntrials * ntime, nchannels))
    train_targets[s] = np.zeros((ntrials * ntime, 1))

    ntrials, nchannels, ntime = train_eeg[s].shape
    for tr in range(0,ntrials):
        for tm in range(0,ntime):
            train_examples[s][tr*ntime+tm,:]= train_eeg[s][tr, :, tm]
            train_targets[s][tr*ntime+tm,0]= train_speech[s][tr,tm]


    eeg_standardizer=  sk.preprocessing.StandardScaler().fit(train_examples[s])
    speech_standardizer = sk.preprocessing.StandardScaler().fit(train_targets[s])







train_examples[s]=eeg_standardizer.transform(train_examples[s])
train_targets[s]=speech_standardizer.transform(train_targets[s])

features= np.concatenate( (train_examples[s], np.square(train_examples[s])) , axis=1)


# ntrials, nchannels, ntime = test_eeg[s].shape
test_examples[s] = np.zeros((1 * ntime, nchannels))
test_targets[s] = np.zeros((1 * ntime, 1))
for tr in range(0, 1):
    for tm in range(0, ntime):
        test_examples[s][tr * ntime + tm, :] = test_eeg[s][tr+1, :, tm]
        test_targets[s][tr * ntime + tm, 0] = test_speech[s][tr+1, tm]

# Make predictions using the testing set

l1_ratio=[0.005,0.01,0.03,0.06,0.1,0.3,0.6,0.7,0.8]
alpha=[0.0005,0.001,0.004,0.008,0.012,0.05,0.08,0.1]

count=0
r2_values=[]

test_examples[s]=eeg_standardizer.transform(test_examples[s])
test_features= np.concatenate( (test_examples[s], np.square(test_examples[s])) , axis=1)

for lr in l1_ratio:
    for a in alpha:

        #LinReg = linear_model.LinearRegression()
        #LinReg.fit(features , train_targets[s])
        elasticNet = linear_model.ElasticNet(l1_ratio=lr,alpha=a)
        elasticNet.fit(features, train_targets[s])



        #speechPrediction1= LinReg.predict(test_features)
        #print('r2 score for lin reg:'+ str(r2_score(speech_standardizer.transform(test_targets[s]),speechPrediction1)))
        speechPrediction2= elasticNet.predict(test_features)
        r2_values.append( r2_score(speech_standardizer.transform(test_targets[s]),speechPrediction2) )


        #plt.figure(0)
        #plt.plot(speechPrediction1)
        #plt.plot(speech_standardizer.transform(test_targets[s]))
        #plt.legend(['prediction','original'])

        plt.figure(count)
        plt.plot(speechPrediction2)
        plt.plot(speech_standardizer.transform(test_targets[s]))
        plt.legend(['Elastic prediction','original'])
        plt.title("alpha="+str(a)+" l1_ratio="+str(lr))
        count=count+1

plt.figure(count)
plt.plot(r2_values)
plt.show()

    # mean_squared_error(test_targets[ntrial,:],speechPrediction)






