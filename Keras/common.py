# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:55:00 2020

@author: Erwing_fc ~erwingforerocastro@gmail.com
"""

import os

root_model_folder='./model/'

if not os.path.exists(root_model_folder):
    os.makedirs(root_model_folder)

model_file_name='{}model.h5'.format(root_model_folder)
model_checkpoint_file_name='{}model-checkpoint.hdf5'.format(root_model_folder)

X_colum_names=['x']
Y_colum_names=['y']