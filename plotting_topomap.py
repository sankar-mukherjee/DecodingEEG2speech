import numpy as np
import matplotlib.pyplot as pp
import mne




data_path = "./ProcessedData/"
eeg="_processed-epo.fif"

T= [-500, -450,-400, -350,-300, -250,-200, -150,-100, -50, 0, 50,100,150, 200,250, 300, 350, 400,450, 500]
info=channels = mne.read_epochs(data_path + 'Alice' + eeg).info
ch=info['ch_names']
delta_speechGAVG = np.load('results_NoTRF/delta_squared/GAVG_sp.npy').squeeze()
delta_lipGAVG = np.load('results_NoTRF/delta_squared/GAVG_lip.npy').squeeze()

a=pp.figure(0)
for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('delta speech delay '+str(T[i]))
 im=mne.viz.plot_topomap(delta_speechGAVG[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False)
 #im=mne.viz.plot_topomap(delta_speechGAVG[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False,names=ch,show_names=True)
a.subplots_adjust(right=0.9)
cbar_ax = a.add_axes([0.92, 0.15, 0.05, 0.7])
a.colorbar(im[0], cax=cbar_ax)
pp.show()

b=pp.figure(0)
for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('delta lips delay = '+str(T[i]))
 im=mne.viz.plot_topomap(delta_lipGAVG[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False)
 #im=mne.viz.plot_topomap(delta_lipGAVG[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False,names=ch,show_names=True)
b.subplots_adjust(right=0.9)
cbar_ax = b.add_axes([0.92, 0.15, 0.05, 0.7])
b.colorbar(im[0], cax=cbar_ax)
pp.show()
