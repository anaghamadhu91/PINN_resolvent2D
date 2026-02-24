#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for complete silence
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sys

run_set = int(sys.argv[1])
run_exp = int(sys.argv[2])
Epochs  = int(sys.argv[3])
kx      = float(sys.argv[4])
omega   = float(sys.argv[5])

np.random.seed(1234)
tf.random.set_seed(1234)

#Loading a plotting from the PINN model
resolvent2D = tf.keras.models.load_model('Saved_model/')

ztest = tf.linspace(-1.,1.,201)
ytest = tf.linspace(0.,2.*np.pi,101)
[Y_test, Z_test] = np.meshgrid(ytest, ztest)

test_data = np.hstack((Y_test.flatten()[:, None], Z_test.flatten()[:, None]))
y_test = tf.cast(test_data[:, 0:1],tf.float32)
z_test = tf.cast(test_data[:, 1:2],tf.float32)

resolvent2D_model = resolvent2D([y_test,z_test])

ur  = np.array(resolvent2D_model[0])[:,0]
vr  = np.array(resolvent2D_model[1])[:,0]
wr  = np.array(resolvent2D_model[2])[:,0]
pr  = np.array(resolvent2D_model[3])[:,0]
ui  = np.array(resolvent2D_model[4])[:,0]
vi  = np.array(resolvent2D_model[5])[:,0]
wi  = np.array(resolvent2D_model[6])[:,0]
pi  = np.array(resolvent2D_model[7])[:,0]
fxr = np.array(resolvent2D_model[8])[:,0]
fyr = np.array(resolvent2D_model[9])[:,0]
fzr = np.array(resolvent2D_model[10])[:,0]
fxi = np.array(resolvent2D_model[11])[:,0]
fyi = np.array(resolvent2D_model[12])[:,0]
fzi = np.array(resolvent2D_model[13])[:,0]

res_u = ur + 1j*ui
res_v = vr + 1j*vi
res_w = wr + 1j*wi
res_p = pr + 1j*pi

frc_x = fxr + 1j*fxi
frc_y = fyr + 1j*fyi
frc_z = fzr + 1j*fzi

Ur = np.real(np.reshape(ur,(len(ztest),len(ytest))))
Vr = np.real(np.reshape(vr,(len(ztest),len(ytest))))
Wr = np.real(np.reshape(wr,(len(ztest),len(ytest))))
Pr = np.real(np.reshape(pr,(len(ztest),len(ytest))))

fig,ax = plt.subplots(2,4,figsize=(19,7))

m = abs(Ur).max()
pl1 = ax[0,0].pcolor(Y_test, Z_test, Ur, cmap='RdBu_r', clim=[-m,m], label = 'U') 
fig.colorbar(pl1, ax=ax[0,0])

m = abs(Vr).max()
pl2 = ax[0,1].pcolor(Y_test, Z_test, Vr, cmap='RdBu_r', clim=[-m,m], label = 'V') 
fig.colorbar(pl2, ax=ax[0,1])

m = abs(Wr).max()
pl3 = ax[0,2].pcolor(Y_test, Z_test, Wr, cmap='RdBu_r', clim=[-m,m], label = 'W') 
fig.colorbar(pl3, ax=ax[0,2])

m = abs(Pr).max()
pl4 = ax[0,3].pcolor(Y_test, Z_test, Pr, cmap='RdBu_r', clim=[-m,m], label = 'P') 
fig.colorbar(pl4, ax=ax[0,3])

history = np.load('history.npz')
epochs = np.linspace(0,Epochs,len(history['residual_loss_ur'])-1); epochs[0] = 100
epochs = np.hstack([[1],epochs])

ax[1,0].plot(epochs, history['residual_loss_ur'], color='black', label='Rur_loss')
ax[1,0].plot(epochs, history['residual_loss_vr'], color='blue', label='Rvr_loss')
ax[1,0].plot(epochs, history['residual_loss_wr'], color='red', label='Rwr_loss')
ax[1,0].plot(epochs, history['residual_loss_wr'], color='green', label='Rcr_loss')
ax[1,0].set_ylabel('Loss', fontsize=16)

ax[1,1].plot(epochs, history['residual_loss_ui'], color='black', label='Rui_loss')
ax[1,1].plot(epochs, history['residual_loss_vi'], color='blue', label='Rvi_loss')
ax[1,1].plot(epochs, history['residual_loss_wi'], color='red', label='Rwi_loss')
ax[1,1].plot(epochs, history['residual_loss_wi'], color='green', label='Rci_loss')

ax[1,2].plot(epochs, history['G_loss1'], color='black', label='G_loss1')
ax[1,2].plot(epochs, history['G_loss2'], color='blue', label='G_loss2')
ax[1,2].set_xlabel('epochs')

ax[1,3].plot(epochs, history['bc_loss'], color='green', label='bc_loss')
ax[1,3].set_xlabel('epochs')

hyperparams = np.load('hyperparams.npz')
title_st = 'beta1='+str(hyperparams['beta1'])+', '+\
'beta2='+str(hyperparams['beta2'])+', '+\
'alpha1='+str(hyperparams['alpha1'])+', '+\
'alpha2='+str(hyperparams['alpha2'])+', '+\
'alpha3='+str(hyperparams['alpha3']) #+',\n '+\
#'Ralpha_ur='+str(hyperparams['Ralpha_ur'])+', '+\
#'Ralpha_ui='+str(hyperparams['Ralpha_ui'])+', '+\
#'Ralpha_vr='+str(hyperparams['Ralpha_vr'])+', '+\
#'Ralpha_vi='+str(hyperparams['Ralpha_vi'])+', '+\
#'Ralpha_wr='+str(hyperparams['Ralpha_wr'])+', '+\
#'Ralpha_wi='+str(hyperparams['Ralpha_wi'])+', '+\
#'Ralpha_cr='+str(hyperparams['Ralpha_cr'])+', '+\
#'Ralpha_ci='+str(hyperparams['Ralpha_ci'])

plt.figtext(0.5, 0.01, title_st, wrap=True, horizontalalignment='center', fontsize=14)

for a in ax.ravel():
    a.legend(loc='upper right')
fig.savefig('Figures_automatic/set%03d_exp%03d.png'%(run_set,run_exp))

#plt.show()
