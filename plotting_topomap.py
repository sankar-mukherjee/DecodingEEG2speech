import numpy as np
import matplotlib.pyplot as pp
import mne




data_path = "./ProcessedData/"
eeg="_processed-epo.fif"

T= [-500, -450,-400, -350,-300, -250,-200, -150,-100, -50, 0, 50,100,150, 200,250, 300, 350, 400,450, 500]
info=channels = mne.read_epochs(data_path + 'Alice' + eeg).info
delta_speechGAVG = np.load('results_NoTRF/delta/GAVG_sp.npy').squeeze()
delta_lipGAVG = np.load('results_NoTRF/delta/GAVG_lip.npy').squeeze()

for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('delta speech delay '+str(T[i]))
 mne.viz.plot_topomap(delta_speechGAVG[:,i], info,vmax=np.amax(delta_speechGAVG),vmin=np.amin(delta_speechGAVG),show=False)
pp.show()

for i in range(0,len(T)):
 pp.subplot(3,7,i+1)
 pp.title('delta lips delay = '+str(T[i]))
 mne.viz.plot_topomap(delta_lipGAVG[:,i], info,vmax=np.amax(delta_lipGAVG),vmin=np.amin(delta_lipGAVG),show=False)
pp.show()
