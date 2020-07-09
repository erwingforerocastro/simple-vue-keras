# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:16:46 2020

@author: Erwing_fc ~erwingforerocastro@gmail.com
"""

from keras.models import load_model

import common

#cargar el estado anterior del modelo
model=load_model(common.model_file_name)

#algunas entradas para predecir

values=[1,2,3,4,5,6,7,8,9,]

#usar el modelo para predecir el precio de la casa

y_predicted=model.predict(values)

for i in range(0,len(values)):
    print(values[i],y_predicted[i])
    


