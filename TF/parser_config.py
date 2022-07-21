#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:47:47 2022

@author: milad
"""

#%%
import configargparse

PARSER = configargparse.ArgParser(default_config_files=['configs/source_8552.config'])
PARSER.add('-c', '--config_path', required=False, is_config_file=True, help='config file path')

# Material properties
PARSER.add('--k', required = True, type = float, help = 'Thermal conductivity (W/(m K))')
PARSER.add('--rho', required = True, type = float, help = 'Density (kg/m3)')
PARSER.add('--Cp', required = True, type = float, help = 'Specific heat capacity (J/ (kg K))')
PARSER.add('--h_b', required = True, type = float, help = 'HTC bottom (W/(m^2 K))')
PARSER.add('--h_t', required = True, type = float, help = 'HTC top (W/(m^2 K))')

# Scale/normalization
PARSER.add('--scale_min', required = True, type = int, help = 'minimum scaled values')
PARSER.add('--scale_max', required = True, type = int, help = 'maximum scaled values')

# Initial, boundary, collocation and labeled points config
PARSER.add('--bc_n', required = True, type = int, help = 'number of points in time for bc')
PARSER.add('--ini_n', required = True, type = int, help = 'number of points for initial condition')
PARSER.add('--L_n', required = True, type = int, help = 'number of points on geometry (collocation)')
PARSER.add('--time_n', required = True, type = int, help = 'number of points in time for PDE (collocation)')
