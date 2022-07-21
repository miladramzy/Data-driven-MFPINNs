#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: https://github.com/miladramzy

Some parts of the code are adopted from:
    https://github.com/saniaki/sequential_PINN
    https://github.com/lululxvi/deepxde
    https://github.com/tims457/ml_notebooks/blob/main/pinns/physics_informed_neural_networks_1.ipynb
"""


#%%
# imports
import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow.keras.backend as kb
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Concatenate, Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Lambda

tf.keras.backend.set_floatx('float64')

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# import raw data

def import_FE_data(direc, fn, part_length, nodes):
    
    raw = pd.read_csv(direc + fn)

    t_np =  np.array(raw['Time'].copy()) # Time (s)
    z_np = np.linspace(0,part_length, num = nodes) # Generate the location (m) of # nodes
    raw_np = np.array(raw)
    Temp = raw_np[:,1:] # Tempe in C (fist column is Time, removed)
    
    u = Temp.flatten('F')[:,None] # flatten Temp C
    u_k = u + 273 # Temp in kelvin
    
    # Domain bounds
    x_lb, x_ub = z_np[0], z_np[-1]
    t_lb, t_ub = t_np[0], t_np[-1]
    
    return z_np, t_np, Temp, u, u_k, x_lb, x_ub, t_lb, t_ub


def Input_data_generator(x_lb, x_ub, x_count, t_lb, t_ub, t_count, scaler, fit=False, internal=False):
    
    x = np.linspace(x_lb, x_ub, x_count)
    t = np.linspace(t_lb, t_ub, t_count)
    count = x.shape[0] * t.shape[0] # number of points
    inp = np.array(np.meshgrid(x, t),dtype=np.float64).T.reshape(-1,2)
    if fit:
        inp_norm = scaler.fit_transform(inp)
    else:
        inp_norm = scaler.transform(inp)
    
    return inp_norm, count, scaler

def internal_data_generator(x_list, t_list, scaler):
    """ generate grid spcace using the x and t lists"""
    x = np.array(x_list)
    t = np.array(t_list)
    count = x.shape[0] * t.shape[0] # number of points
    inp = np.array(np.meshgrid(x, t),dtype=np.float64).T.reshape(-1,2)
    inp_norm = scaler.transform(inp)
    return inp_norm, count

# initilizaers 
he = tf.keras.initializers.he_uniform()
random_uniform = tf.keras.initializers.RandomUniform()


def pred_NN(input_shape, hidden_layers, mode):
    
    x_inputs = Input(shape=input_shape, name='x_inputs')
    t_inputs = Input(shape=input_shape, name='t_inputs')
    ylf_inputs = Input(shape=input_shape, name='ylf_inputs')
    yhf_inputs = Input(shape=input_shape, name='yhf_inputs')    
    
    nodes = 30
    if mode =='PINN':
        h = Concatenate(axis=1)([x_inputs, t_inputs])
    elif mode == 'MPINN':
        h = Concatenate(axis=1)([x_inputs, t_inputs, ylf_inputs])
    
    for i in np.arange(hidden_layers):
        h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    
    output = Dense(1, kernel_initializer=random_uniform, activation='softplus', name='out_Temp')(h)
    
    model = Model(inputs = [x_inputs, t_inputs, ylf_inputs, yhf_inputs],
                  outputs = [output])
    
    return model


def Pred_PINN(input_shape):
    '''
    model input is the time 't' and location 'x' 
    '''
    x_inputs = Input(shape=input_shape, name='x_inputs')
    t_inputs = Input(shape=input_shape, name='t_inputs')
    ylf_inputs = Input(shape=input_shape, name='ylf_inputs')
    yhf_inputs = Input(shape=input_shape, name='yhf_inputs')
    
    
    nodes = 30
    h = Concatenate(axis=1)([x_inputs, t_inputs])
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    
    # degree of cure output
    output = Dense(1, kernel_initializer=random_uniform, activation='softplus', name='out_Temp')(h)
    
    
    model = Model(inputs = [x_inputs, t_inputs, ylf_inputs, yhf_inputs],
                  outputs = [output])
    
    return model

def MF_PINN(input_shape):
    '''
    model input is the time 't', location 'x' and lf temperature 'y_lf'
    '''
    x_inputs = Input(shape=input_shape, name='x_inputs')
    t_inputs = Input(shape=input_shape, name='t_inputs')
    ylf_inputs = Input(shape=input_shape, name='ylf_inputs')
    yhf_inputs = Input(shape=input_shape, name='yhf_inputs')
    
    
    nodes = 30
    h = Concatenate(axis=1)([x_inputs, t_inputs, ylf_inputs])
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    h = Dense(nodes, kernel_initializer=he, activation='tanh')(h)
    
    # degree of cure output
    output = Dense(1, kernel_initializer=random_uniform, activation='softplus', name='out_Temp')(h)
    
    
    model = Model(inputs = [x_inputs, t_inputs, ylf_inputs, yhf_inputs],
                  outputs = [output])
    
    return model


def MF_DNN(input_shape):
    he = tf.keras.initializers.he_uniform()
    random_uniform = tf.keras.initializers.RandomUniform()
    
    activation = 'selu'
    nodes = 30
    
    inputs = Input(shape=(input_shape), name="input")
    h = Dense(nodes, kernel_initializer=he, activation=activation)(inputs)
    h = Dense(nodes, kernel_initializer=he, activation=activation)(h)
    h = Dense(nodes, kernel_initializer=he, activation=activation)(h)
    h = Dense(nodes, kernel_initializer=he, activation=activation)(h)
    h = Dense(nodes, kernel_initializer=he, activation=activation)(h)
    outputs = Dense(1, kernel_initializer=random_uniform)(h)
    
    model = Model(inputs=inputs, outputs = outputs)
    
    return model

""" ************** Network losses ***************"""

def data_L_func(LambdaList):

    output_T = LambdaList[2]
    yhf = LambdaList[3]
    
    XX = tf.math.sign(yhf)
    
    return (yhf - output_T)*XX

# Make bottom boundary condition loss
def make_bc_b_func(T_bc_func, scaler1, T_scale, scale_min, k, h_b):
    def bc_b_func(LambdaList):
        x_inp = LambdaList[0]
        t_inp = LambdaList[1]
        output_T = LambdaList[2]
        

        T_bc_b = T_bc_func(t_inp , scaler1)
        T_bc_b = T_bc_b / T_scale
        
        # XX=1 where loss is applicable, XX=0 where loss is not applicable
        XX = 0.5 * (1.0-tf.math.sign(x_inp-scale_min-1e-15)) 
            
        Tx = tf.gradients(output_T, x_inp,unconnected_gradients='zero')
        L = ((T_bc_b - output_T) + k / h_b * Tx[0] * scaler1[0,0]) * XX 
            
        return L
    return bc_b_func

def make_bc_t_func(T_bc_func, scaler1, T_scale, scale_max, k, h_t):
    def bc_t_func(LambdaList):
        
        x_inp = LambdaList[0]
        t_inp = LambdaList[1]
        output_T = LambdaList[2]
        
        T_bc_t = T_bc_func(t_inp , scaler1)
        T_bc_t = T_bc_t / T_scale
       
        # XX=1 where loss is applicable, XX=0 where loss is not applicable
        XX = 0.5 * (1.0+tf.math.sign(x_inp-scale_max+1e-15))
        
        Tx = tf.gradients(output_T, x_inp,unconnected_gradients='zero')
        L = ((T_bc_t - output_T) - k / h_t * Tx[0]  * scaler1[0,0])  * XX 

        return L
    return bc_t_func

def make_initial_func(T_ini, T_scale, scale_min, scale_max):
    def initial_func(LambdaList):
        
        x_inp = LambdaList[0]
        t_inp = LambdaList[1]
        output_T = LambdaList[2]
        
        T_ini_arr = t_inp * 0 + T_ini / T_scale
        
        # XX=1 where loss is applicable, XX=0 where loss is not applicable
        XX = 0.5 * (1.0-tf.math.sign(t_inp-scale_min-1e-10)) \
           * 0.5 * (1.0+tf.math.sign(x_inp-scale_min-1e-10))  \
           * 0.5 * (1.0 - tf.math.sign(x_inp-scale_max+1e-10)) 
        
        L = (output_T - T_ini_arr) * XX
       
        return L
    return initial_func

def make_pde_func(scale_min, scale_max, a_c_normalized):
    def pde_func(LambdaList):
        x_inp = LambdaList[0]
        t_inp = LambdaList[1]
        out_T = LambdaList[2]
        
        def replace_none_with_zero(l):
            return [0 if i==None else i for i in l]
        
        # gradients
        Tt = tf.gradients(out_T, t_inp,unconnected_gradients='zero')
        Tx = tf.gradients(out_T, x_inp,unconnected_gradients='zero')
        Txx = tf.gradients(Tx[0], x_inp, unconnected_gradients='zero')
        
        # XX=1 where loss is applicable, XX=0 where loss is not applicable
        XX = 0.5 * (1.0+tf.math.sign(t_inp-scale_min-1e-15)) \
           * 0.5 * (1.0+tf.math.sign(x_inp-scale_min-1e-15)) \
           * 0.5 * (1.0-tf.math.sign(x_inp-scale_max+1e-15)) 
        
        L = (Tt[0] - a_c_normalized * Txx[0]) * XX
                  
        return L
    return pde_func


""" *****  Gradient pathology loss weight ***** """

# Calculate loss grads
@tf.function
def loss_grads(model, input_data_1, input_data_2, input_data_3, input_data_4):
    
    losses = model(inputs = [input_data_1, input_data_2, input_data_3, input_data_4])
    mse_losses = [tf.reduce_mean(li**2) for li in losses]
    L_grads = [tf.gradients(i, model.trainable_variables)  for i in mse_losses]
    trainable_count = tf.cast(model.count_params(), tf.float64) 
    
    return L_grads, trainable_count

# Find mean loss grad and calculate new alpha
def mean_of_loss_grads(L_grads, pde_grad_max, lambda_old, trainable_count):
    
    layer_mean = [tf.reduce_mean(abs(i)) for i in L_grads]
    net_mean= sum(layer_mean) / len(layer_mean)
    lambda_i = pde_grad_max / (net_mean * lambda_old)
    
    return lambda_i
    
def update_weights(lambda_i_old, lambda_i):
    
    lambda_i_new =  0.9 *lambda_i_old + (1-0.9) * lambda_i
    
    return lambda_i_new


# callbck class for Temp adaptive loss weight

class Adaptive_loss(tf.keras.callbacks.Callback):
    def __init__(self,pde_L_w, initial_L_w, bc_b_L_w, bc_t_L_w, data_L_w, pde_L_w_hist, initial_L_w_hist, bc_b_L_w_hist, bc_t_L_w_hist, data_L_w_hist, input_train_x_np, input_train_t_np, y_lf_np, Heat_model):

        self.pde_L_w = pde_L_w
        self.initial_L_w = initial_L_w
        self.bc_b_L_w = bc_b_L_w
        self.bc_t_L_w = bc_t_L_w
        self.data_L_w = data_L_w
        
        self.pde_L_w_hist = pde_L_w_hist
        self.initial_L_w_hist = initial_L_w_hist
        self.bc_b_L_w_hist = bc_b_L_w_hist
        self.bc_t_L_w_hist = bc_t_L_w_hist
        self.data_L_w_hist = data_L_w_hist
        
        self.input_train_x_np = input_train_x_np
        self.input_train_t_np = input_train_t_np
        self.y_lf_np = y_lf_np
        
        self.Heat_model = Heat_model
        
    def on_epoch_end(self, epoch, logs={}):
        
        if epoch%10 == 0 and epoch > 0:
            
            # Calculate the grad for each loss function
            grads_all, trainable_count = loss_grads(self.Heat_model, self.input_train_x_np, self.input_train_t_np, self.y_lf_np, self.y_lf_np)
            grads_of_pde, grads_of_iniT_L, grads_of_bc_b_L, grads_of_bc_t_L = grads_all
            
            # Calculate the max pde grad value
            #max_of_pde_grads = max_of_loss_grads(grads_of_pde)
            max_of_pde_grads = tf.reduce_max([tf.reduce_max(abs(i)) for i in grads_of_pde])
            pde_L_w_new = self.pde_L_w
            
            # Initial condition weight update
            iniT_loss_lambda = mean_of_loss_grads(grads_of_iniT_L, max_of_pde_grads, self.initial_L_w, trainable_count)
            initial_L_w_new = update_weights(self.initial_L_w , iniT_loss_lambda)

            # Boundary condition - bottom weight update            
            bc_b_loss_lambda = mean_of_loss_grads(grads_of_bc_b_L, max_of_pde_grads, self.bc_b_L_w, trainable_count)
            bc_b_L_w_new = update_weights(self.bc_b_L_w , bc_b_loss_lambda)            
            
            
            # Boundary condition - top weight update    
            bc_t_loss_lambda = mean_of_loss_grads(grads_of_bc_t_L, max_of_pde_grads, self.bc_t_L_w, trainable_count)
            bc_t_L_w_new = update_weights(self.bc_t_L_w , bc_t_loss_lambda)  
            
            # Update weights, append history and print
            kb.set_value(self.initial_L_w, initial_L_w_new)
            kb.set_value(self.bc_b_L_w, bc_b_L_w_new)
            kb.set_value(self.bc_t_L_w, bc_t_L_w_new)
            
            self.pde_L_w_hist.extend([pde_L_w_new])
            self.initial_L_w_hist.extend([initial_L_w_new])
            self.bc_b_L_w_hist.extend([bc_b_L_w_new])
            self.bc_t_L_w_hist.extend([bc_t_L_w_new])
            
            tf.print('--------------------------')
            tf.print('Loss weights are updated to:')
            tf.print(self.pde_L_w, self.initial_L_w, self.bc_b_L_w, self.bc_t_L_w, self.data_L_w)
            tf.print('--------------------------')

