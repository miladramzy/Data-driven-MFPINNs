
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:02:30 2021

@author: @miladramzy

Some parts of the code are adopted from:
    https://github.com/saniaki/sequential_PINN
    https://github.com/lululxvi/deepxde
    https://github.com/tims457/ml_notebooks/blob/main/pinns/physics_informed_neural_networks_1.ipynb
"""

#%%
# imports

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, Lambda

from parser_config import PARSER

tf.keras.backend.set_floatx('float64')

#from Source_Training import 
from PINNs_network import *

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Print GPU info
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%
"""STEP 1: Load HF data"""

# Import FE data
direc = r'/data/'
fn = 'target.csv'
z_np, t_np, T, u, u_k, x_lb, x_ub, t_lb, t_ub = import_FE_data(direc, fn, 0.02, 41)

args = PARSER.parse_args(['--config_path', 'configs/config.config'])
# Specifiy material properties
k = args.k # (W/(m K))
rho = args.rho # (kg/m3)
Cp = args.Cp # (J/ (kg K))
h_b = args.h_b # (W/(m^2 K))
h_t = args.h_t # (W/(m^2 K))
a =  k/(rho*Cp) # Thermal diffusivity in heat trasnfer PDE (m2 / s)

# feature scaling (here, 0 and 1 were chosen)
scale_min = args.scale_min # minimum scaled value
scale_max = args.scale_max # maximum scaled value
min_max=MinMaxScaler(feature_range=(scale_min, scale_max))

""" Location and time for labeled internal points """
x_int = z_np[2::7]
t_int = t_np[[-25, -20, -15, -10, -5]]

"""Specifying the number of initial, boundary and PDE data"""
n_t_bc = args.bc_n # number of points in time for bc
n_x_pde = args.time_n # number of points on geometry
n_t_pde = args.L_n # number of points in time for PDE
n_ini = args.ini_n # number of points for initial condition

"""Cure cycle Specs"""
T_ini = tf.cast(0+273, tf.float64) # initial temperature (k)
time_ramp_e = tf.cast(600, tf.float64) # heat ramp ends at (s)
time_hold_e = tf.cast(1800, tf.float64) # hold ends at (s)
ramp_rate = tf.cast(50/600, tf.float64) # K/s
T_hold = tf.cast(50 + 273, tf.float64) # Temperature at hold (k)
Temp_k = u + 273 # Test Temperature (k)

"""Generate input data"""
# bc points
input_bc_np_norm, bc_count, scaler = Input_data_generator(x_lb, x_ub, 2, t_lb, t_ub, n_t_bc, min_max, fit=True)
# initial points
input_init_np_norm, init_count, _ = Input_data_generator(x_lb, x_ub, n_ini, t_lb, t_lb, 1, scaler, fit=False)
# Collocation points
input_pde_np_norm, pde_count, _ = Input_data_generator(x_lb, x_ub, n_x_pde, t_lb, t_ub, n_t_pde, scaler, fit=False)
# Internal points
input_internal_np_norm, internal_count = internal_data_generator(x_int, t_int, scaler)

scaler1 = np.array([1,1]).reshape(1, 2) 
scaler1 = min_max.transform(scaler1)
# Normalized  diffusivity
a_c_normalized = a * ((scaler1[0,0])**2)/scaler1[0,1]

""" Output y """
T_scale = np.max(Temp_k) 

x_internal = z_np
t_internal = t_np
input_internal= np.array(np.meshgrid(x_internal, t_internal),dtype=np.float64).T.reshape(-1,2)
input_internal_norm = scaler.transform(input_internal)

hf_pd = pd.DataFrame(np.hstack([input_internal_norm, Temp_k/T_scale])
                     , columns=['x', 't', 'y'])

data_nonnorm = np.hstack([input_internal, u])

# internal_pd creates a dataframe that stores HF outputs along with their corrosponding x and t. This can
# then be used in the y_HF loss function. We add internal_pd['y'] to y_all_hf.
internal_pd = pd.DataFrame()
for i in np.arange(input_internal_np_norm.shape[0]):
    x = input_internal_np_norm[i,0]
    t = input_internal_np_norm[i,1]
    cond = hf_pd[((np.isclose(hf_pd.x,x,atol=1e-03)) & (np.isclose(hf_pd.t,t, atol=1e-03, rtol=1e-03)))]
    if cond.shape[0]>0:
        internal_pd = internal_pd.append(cond)

input_internal_np_norm = np.array(internal_pd[['x','t']])
# Combine all input data
input_all_np_norm = kb.concatenate((input_pde_np_norm, input_bc_np_norm, input_init_np_norm, input_internal_np_norm), axis=0)
input_x_norm = input_all_np_norm[:,0]
input_t_norm = input_all_np_norm[:,1]

# total count
data_num = bc_count + init_count + pde_count + internal_pd.shape[0]
data_num_pinns = bc_count + init_count + pde_count

# Total y
target_data_all_np = np.zeros((data_num,1))
y_train = target_data_all_np
y_all_hf = np.vstack([np.zeros((data_num_pinns,1)), np.reshape(np.array(internal_pd['y']), (internal_pd['y'].shape[0], 1))])

# Boundary temperature function used in BC loss functions
def T_bc_func(t_inp, scaler):
    
    t_inp = t_inp / tf.cast(scaler[0,1], tf.float64)
    T_ev = tf.where(t_inp < time_ramp_e, T_ini + t_inp*ramp_rate, 
                    tf.where(t_inp > time_hold_e, T_hold - (t_inp-time_hold_e)*ramp_rate, T_hold))
    T_ev = tf.reshape(T_ev, [tf.shape(T_ev)[0],1])
    
    return T_ev

#%%
"""STEP 2: Load LF model, generate y_LF"""
# y low-fidelity

tf.keras.backend.clear_session()
source_model = pred_NN(input_shape=(1), hidden_layers=9, mode='PINN')

source_model.load_weights('models/source_model')

y_lf = source_model.predict([input_x_norm,input_t_norm,input_t_norm,input_t_norm])

#%%
"""STEP 3: Train MPINN"""

""" ************** 3.1. hf network ***************"""

# initilizaers 
he = tf.keras.initializers.he_uniform()
random_uniform = tf.keras.initializers.RandomUniform()
tanh = tf.keras.activations.tanh


def Temp_PINN(input_shape, data=False, lf = False):
    '''
    model inputs are time 't', location 'x' and low-fidelity temperature 'y_lf'
    '''
    x_inputs = Input(shape=input_shape, name='x_inputs')
    t_inputs = Input(shape=input_shape, name='t_inputs')
    ylf_inputs = Input(shape=input_shape, name='ylf_inputs')
    yhf_inputs = Input(shape=input_shape, name='yhf_inputs')
    
    nodes = 30
    
    if lf:
        h = Concatenate(axis=1)([x_inputs, t_inputs, ylf_inputs])
    else:
        h = Concatenate(axis=1)([x_inputs, t_inputs])
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    h = Dense(nodes, kernel_initializer=he, activation=tanh)(h)
    
    # degree of cure output
    output = Dense(1, kernel_initializer=random_uniform, activation='softplus', name='out_Temp')(h)
        
    # losses
    pde_L = Lambda(pde_func, name='pde_L')([x_inputs, t_inputs, output]) # PDE loss
    initial_L = Lambda(initial_func, name='initial_L')([x_inputs, t_inputs, output]) # initial condition loss
    bc_b_L = Lambda(bc_b_func, name='bc_b_L')([x_inputs, t_inputs, output]) # bc bottom loss 
    bc_t_L = Lambda(bc_t_func, name='bc_t_L')([x_inputs, t_inputs, output]) # bc top loss 
    
    
    if data:
        data_L = Lambda(data_L_func, name='data_L')([x_inputs, t_inputs, output, yhf_inputs])
        
        model = Model(inputs = [x_inputs, t_inputs, ylf_inputs, yhf_inputs],
                      outputs = [pde_L, initial_L, bc_b_L, bc_t_L, data_L])
    else:
        model = Model(inputs = [x_inputs, t_inputs, ylf_inputs, yhf_inputs],
                      outputs = [pde_L, initial_L, bc_b_L, bc_t_L])        
    
    return model

tot_num = data_num

#%%
""" ************** 3.2. Network losses ***************"""

bc_b_func = make_bc_b_func(T_bc_func, scaler1, T_scale, scale_min, k, h_b)
bc_t_func = make_bc_t_func(T_bc_func, scaler1, T_scale, scale_max, k, h_t)
initial_func = make_initial_func(T_ini, T_scale, scale_min, scale_max)
pde_func = make_pde_func(scale_min, scale_max, a_c_normalized)

#%%
"""******************** 3.3. Training *********************"""

# Set to "True" if the labeled data in the domain is included for the training
Data_mode = False

tf.keras.backend.clear_session()
Heat_model = Temp_PINN(input_shape=(1), data=Data_mode, lf = True)

import time
start_time = time.time()

# finding the layer number of Temp output in Heat model
out_T_indx = None
for idx, layer in enumerate(Heat_model.layers):
    if layer.name == 'out_Temp':
        out_T_indx = idx
        break

epochs = 100

# Initial weights
pde_L_w = kb.variable(1.0)
initial_L_w = kb.variable(1.0)
bc_b_L_w = kb.variable(1.0)
bc_t_L_w = kb.variable(1.0)
data_L_w = kb.variable(1.0)

opt_T = tf.keras.optimizers.Adam(learning_rate=1e-3)

# compile
Heat_model.compile(loss=['mse', 'mse', 'mse', 'mse', 'mse'],optimizer=opt_T,
                   loss_weights = [pde_L_w, initial_L_w, bc_b_L_w, bc_t_L_w, data_L_w])

patience_T = 20
reduce_lr_T = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                 factor=0.5, patience=patience_T,
                                                 verbose=0, mode='auto',
                                                 min_delta=0.0001, cooldown=0, min_lr=1e-8)
batch_size_T = 64

pde_L_w_hist = []
initial_L_w_hist = []
bc_b_L_w_hist = []
bc_t_L_w_hist = []
data_L_w_hist = []

input_train_x_np = np.reshape(input_x_norm, (input_x_norm.shape[0], 1))
input_train_t_np = np.reshape(input_t_norm, (input_t_norm.shape[0], 1))
y_lf_np = y_all_hf

if Data_mode:
    history_Heat = Heat_model.fit({'x_inputs':input_x_norm,'t_inputs':input_t_norm,
                                   'ylf_inputs':y_lf, 'yhf_inputs':y_all_hf},
                                  {'pde_L':y_train,
                                   'initial_L':y_train, 'bc_b_L':y_train,
                                   'bc_t_L':y_train, 'data_L':y_train}, 
                                  batch_size=batch_size_T,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[reduce_lr_T,Adaptive_loss(pde_L_w, initial_L_w, bc_b_L_w, bc_t_L_w, data_L_w,
                                                                       pde_L_w_hist, initial_L_w_hist, bc_b_L_w_hist, bc_t_L_w_hist, data_L_w_hist,
                                                                       input_train_x_np, input_train_t_np, y_lf_np, Heat_model)])

else:
    # without data
    history_Heat = Heat_model.fit({'x_inputs':input_x_norm,'t_inputs':input_t_norm,
                                   'ylf_inputs':y_lf, 'yhf_inputs':y_all_hf},
                                  {'pde_L':y_train,
                                   'initial_L':y_train, 'bc_b_L':y_train,
                                   'bc_t_L':y_train}, 
                                  batch_size=batch_size_T,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[reduce_lr_T,Adaptive_loss(pde_L_w, initial_L_w, bc_b_L_w, bc_t_L_w, data_L_w,
                                                                       pde_L_w_hist, initial_L_w_hist, bc_b_L_w_hist, bc_t_L_w_hist, data_L_w_hist,
                                                                       input_train_x_np, input_train_t_np, y_lf_np, Heat_model)])


for i in history_Heat.history.keys():
    plt.plot(history_Heat.history[i], label = i)
plt.yscale('logit')
plt.legend()
plt.show()

end_time = time.time()
print(end_time - start_time)

"""******************** 3.4. Evaluation *********************"""
# Mean relative l2 norm
t = t_np
x = z_np

test_num = x.shape[0] * t.shape[0] # number of collocation points
test_data_np = np.array(np.meshgrid(x, t),dtype=np.float64).T.reshape(-1,2)

test_data_np = min_max.transform(test_data_np) # Normalize to [0,1]: good for model input
test_x = test_data_np[:,0]
test_t = test_data_np[:,1]
test_x = np.reshape(test_x, (test_x.shape[0], 1))
test_t = np.reshape(test_t, (test_t.shape[0], 1))

y_lf_true = source_model.predict([test_x,test_t,test_t,test_t])

# Network prediction setup
out_T_indx = None
for idx, layer in enumerate(Heat_model.layers):
    if layer.name == 'out_Temp':
        out_T_indx = idx
        break
inp = Heat_model.input  
functors = kb.function([inp], Heat_model.layers[out_T_indx].output)
predictions_test_T = functors([test_x, test_t, y_lf_true, y_lf_true])

predictions_test_T_arr = tf.reshape(predictions_test_T[0:test_num,:], [x.shape[0], t.shape[0]]) * T_scale - 273
error_ed = np.absolute(predictions_test_T_np - T.T)

err_1, err_2 = error_ed.shape
error_1d = np.reshape(error_ed, [err_1*err_2, -1])
numer = np.sum(error_1d**2)

T_1d = np.reshape(T, [err_1*err_2, -1])
denom = np.sum(T_1d**2)

rel_l2 = np.sqrt(numer/denom)
print(rel_l2)