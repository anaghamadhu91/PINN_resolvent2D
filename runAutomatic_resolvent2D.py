import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' for complete silence
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from keras.initializers import GlorotUniform

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.chdir('./')

#tf.config.threading.set_intra_op_parallelism_threads(16)
#tf.config.threading.set_inter_op_parallelism_threads(1)

import sys

from resolvent2D_automatic import PdeModel, get_ibc_and_inner_data

run_set = int(sys.argv[1])
run_exp = int(sys.argv[2])
hyperparam_filename = 'hyperparams/hyperparam_set%03d_exp.dat'%run_set
hyperparam_expname = 'exp%03d'%run_exp

np.random.seed(1234)
tf.random.set_seed(1234)

# Data PreProcessing
num_samples = 2000   #number of sample points
Re = 1000.          #kinematic viscosity
kx = 0.
omega = 0.

#boundary and grid data
yb_top, zb_top, yb_bottom, zb_bottom, yb_left, zb_left, yb_right, zb_right, yd, zd, \
            ur, vr, wr, ui, vi, wi, fxr, fyr, fzr, fxi, fyi, fzi = get_ibc_and_inner_data(num_samples=num_samples)

#inputs and outputs
ivals = {'yin': yd, 'zin': zd, 'yb_top': yb_top, 'zb_top': zb_top, 'yb_bottom': yb_bottom, 'zb_bottom': zb_bottom, 'yb_left': yb_left, 'zb_left': zb_left, 'yb_right': yb_right, 'zb_right': zb_right }
ovals = {'urb':ur, 'vrb':vr, 'wrb':wr, 'uib':ui, 'vib':vi, 'wib':wi, 'fxrb':fxr, 'fyrb':fyr, 'fzrb':fzr, 'fxib':fxi, 'fyib':fyi, 'fzib':fzi}
parameters = {'Re': Re, 'kx': kx, 'omega':omega}
initializer = GlorotUniform(seed = 1234)

#model parameters
input1 = layers.Input(shape=(1,), name='y_input')
input2 = layers.Input(shape=(1,), name='z_input')
x = layers.Concatenate()([input1, input2])
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer)(x)

ur_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
vr_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
wr_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
pr_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
ui_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
vi_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
wi_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
pi_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
fxr_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
fyr_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
fzr_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
fxi_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
fyi_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)
fzi_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer)(x)

x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_1')(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_2')(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_3')(x)
x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_4')(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_5')(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_6')(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_7')(x)
#x = layers.Dense(units=128, activation='swish', kernel_initializer=initializer, name='Hidden_layer_8')(x)

ur_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_ur')(x)
vr_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_vr')(x)
wr_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_wr')(x)
pr_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_pr')(x)
ui_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_ui')(x)
vi_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_vi')(x)
wi_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_wi')(x)
pi_layer  = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_pi')(x)
fxr_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_fxr')(x)
fyr_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_fyr')(x)
fzr_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_fzr')(x)
fxi_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_fxi')(x)
fyi_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_fyi')(x)
fzi_layer = layers.Dense(units=64, activation='swish', kernel_initializer=initializer, name='Layer_fzi')(x)

our  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_ur')(ur_layer)
ovr  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_vr')(vr_layer)
owr  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_wr')(wr_layer)
opr  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_pr')(pr_layer)
oui  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_ui')(ui_layer)
ovi  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_vi')(vi_layer)
owi  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_wi')(wi_layer)
opi  = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_pi')(pi_layer)
ofxr = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_fxr')(fxr_layer)
ofyr = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_fyr')(fyr_layer)
ofzr = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_fzr')(fzr_layer)
ofxi = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_fxi')(fxi_layer)
ofyi = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_fyi')(fyi_layer)
ofzi = layers.Dense(units=1, use_bias=False, kernel_initializer=initializer, name='output_fzi')(fzi_layer)

model = keras.Model([input1, input2], [our, ovr, owr, opr, oui, ovi, owi, opi, ofxr, ofyr, ofzr, ofxi, ofyi, ofzi])

model.summary()

# Training the model
loss_fn = keras.losses.MeanSquaredError()

#import numpy as np
#import matplotlib.pyplot as plt
#t = np.linspace(0,20000,100)
#lr = 1e-3 * (0.9)**(t/1000)
#lr_pc = [8e-4 * (t0<=2000) + 5e-4 * (t0>2000 and t0<=6000) + 2e-4 * (t0>6000 and t0<=15000) + 1e-4 * (t0>15000) for t0 in t]
#plt.plot(t, lr); plt.plot(t, np.asarray(lr_pc)); plt.show()

initial_learning_rate = 1e-3
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
     initial_learning_rate=initial_learning_rate,
     decay_steps=1000,
     decay_rate=0.9,
     staircase=False)
#lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
#    boundaries=[2000,6000,15000],          # after 2000 epochs
#    values=[8e-4, 5e-4, 2e-4, 1e-4]         # LR before 2000, LR after 2000
#)
optimizer = keras.optimizers.legacy.Adam(learning_rate=lr_schedule, beta_1 = 0.2, beta_2 = 0.7)

model_dict = {"nn_model": model}

metrics = { "loss": keras.metrics.Mean(name='loss'),
            "residual_loss": keras.metrics.Mean(name='residual_loss'),
            "residual_loss_ur": keras.metrics.Mean(name='residual_loss_ur'),
            "residual_loss_ui": keras.metrics.Mean(name='residual_loss_ui'),
            "residual_loss_vr": keras.metrics.Mean(name='residual_loss_vr'),
            "residual_loss_vi": keras.metrics.Mean(name='residual_loss_vi'),
            "residual_loss_wr": keras.metrics.Mean(name='residual_loss_wr'),
            "residual_loss_wi": keras.metrics.Mean(name='residual_loss_wi'),
            "residual_loss_cr": keras.metrics.Mean(name='residual_loss_cr'),
            "residual_loss_ci": keras.metrics.Mean(name='residual_loss_ci'),
            "G_loss": keras.metrics.Mean(name='G_loss'),
            "G_loss1": keras.metrics.Mean(name='G_loss1'),
            "G_loss2": keras.metrics.Mean(name='G_loss2'),
            "bc_loss": keras.metrics.Mean(name='bc_loss') }

cm = PdeModel(inputs=ivals, outputs=ovals, get_models=model_dict, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
              parameters=parameters, hyperparam_filename=hyperparam_filename, hyperparam_expname=hyperparam_expname, ibc_layer=False, mask=None)

Epochs=10000
log_dir = 'log_output/'
history = cm.run(epochs=Epochs, proj_name="resolvent1D_channel", log_dir=log_dir,
                 wb=False, verbose_freq=1000)

# Evaluation
cm.nn_model.save('Saved_model/', save_format='tf')
np.savez('history.npz', **history)
os.system("python Post_Processing2D_automatic.py %03d %03d %06d %.4f %.4f"%(run_set,run_exp,Epochs,kx,omega))
