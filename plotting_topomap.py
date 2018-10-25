import numpy as np
import matplotlib.pyplot as pp
import mne




data_path = "./ProcessedData/"
eeg="_processed-epo.fif"

T= [-500, -450,-400, -350,-300, -250,-200, -150,-100, -50, 0, 50,100,150, 200,250, 300, 350, 400,450, 500]
info=channels = mne.read_epochs(data_path + 'Alice' + eeg).info
ch=info['ch_names']
delta_speechGAVG = np.load('results_RidgeReg\single_channels\delta\GAVG_sp_are_Time_band_pass_derivative1_yesNeighbours_noShuffled.npy').squeeze()
delta_lipGAVG = np.load('results_RidgeReg\single_channels\delta\GAVG_lip_are_Time_band_pass_derivative1_yesNeighbours_noShuffled.npy').squeeze()

delta_speechGAVG_shuffled = np.load('results_RidgeReg\single_channels\delta\GAVG_sp_are_Time_band_pass_derivative1_yesNeighbours_yesShuffled.npy').squeeze()
delta_lipGAVG_shuffled = np.load('results_RidgeReg\single_channels\delta\GAVG_lip_are_Time_band_pass_derivative1_yesNeighbours_yesShuffled.npy').squeeze()

r=0
delta_speechGAVG=delta_speechGAVG[:,:,r]
delta_lipGAVG=delta_lipGAVG[:,:,r]
delta_speechGAVG_shuffled=delta_speechGAVG_shuffled[:,:,r]
delta_lipGAVG_shuffled=delta_lipGAVG_shuffled[:,:,r]


a=pp.figure(0)
for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('delta speech delay '+str(T[i]))
 im=mne.viz.plot_topomap(delta_speechGAVG[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False)
 #im=mne.viz.plot_topomap(delta_speechGAVG[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False,names=ch,show_names=True)
a.subplots_adjust(right=0.9)
cbar_ax = a.add_axes([0.92, 0.15, 0.05, 0.7])
a.colorbar(im[0], cax=cbar_ax)

a1=pp.figure(1)
for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('random speech delay '+str(T[i]))
 im1=mne.viz.plot_topomap(delta_speechGAVG_shuffled[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False)
 #im=mne.viz.plot_topomap(delta_speechGAVG[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False,names=ch,show_names=True)
a1.subplots_adjust(right=0.9)
cbar_ax = a1.add_axes([0.92, 0.15, 0.05, 0.7])
a1.colorbar(im1[0], cax=cbar_ax)

pp.show()

b=pp.figure(2)
for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('delta lips delay = '+str(T[i]))
 im=mne.viz.plot_topomap(delta_lipGAVG[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False)
 #im=mne.viz.plot_topomap(delta_lipGAVG[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False,names=ch,show_names=True)
b.subplots_adjust(right=0.9)
cbar_ax = b.add_axes([0.92, 0.15, 0.05, 0.7])
b.colorbar(im[0], cax=cbar_ax)

b1=pp.figure(3)
for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('random lips delay = '+str(T[i]))
 im1=mne.viz.plot_topomap(delta_lipGAVG_shuffled[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False)
 #im=mne.viz.plot_topomap(delta_lipGAVG[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False,names=ch,show_names=True)
b1.subplots_adjust(right=0.9)
cbar_ax = b1.add_axes([0.92, 0.15, 0.05, 0.7])
b1.colorbar(im1[0], cax=cbar_ax)
pp.show()
