# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:16:46 2020

@author: Erwing_fc ~erwingforerocastro@gmail.com
"""
#importamos las liberias
import pandas as pd
import matplotlib.pyplot as plt

#tensorflow
import tensorflow as tf
from tensorflow.python.client import device_lib

#sklearn
from sklearn.model_selection import train_test_split

#keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping,ModelCheckpoint

import common

print(tf.__version__)
print(device_lib.list_local_devices())

#cargar el dataset

CSV_PATH="./dataset/dataset.csv"
df=pd.read_csv(CSV_PATH,index_col=False)

#evaluamos el contenido del dataset

print(df.head())
print(df.columns)

X=df[common.X_colum_names]
Y=df[common.Y_colum_names]

print(X.head(),Y.head())

#usamos sklearn para separar los grupos de entrenamiento y test

X_train,Y_train,X_test,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

print("Dataset tamaño de entrenamiento: \n",len(X_train))
print(X_train.head(),Y_train.head())
print("Dataset tamaño de test: \n",len(X_test))
print(X_test.head(),Y_test.head())

#construimos el modelo (1 capa)

def build_model(x_size,y_size):
    t_model=Sequential()
    t_model.add(Dense(x_size,input_shape=(x_size,)))
    t_model.compile(loss="mean_squared_error",optimizer="sgd",metrics=[metrics.mae])
    return (t_model)

print(X_train.shape[1],Y_train.shape[1])

#observamos el modelo creado
model=build_model(X_train.shape[1],Y_train.shape[1])
model.summary()

#configurar el entrenamiento

epochs=50 #ciclos de entrenamiento
batch_size=30 #tamaño del lote

keras_callbacks=[
ModelCheckpoint(common.model_checkpoint_file_name,
                monitor='val_mean_absolute_error',
                save_best_only=True,
                verbose=0),
EarlyStopping(monitor='val_mean_absolute_error',patience=20,verbose=2)
]


#entrenar el modelo

history=model.fit(X_train,Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  verbose=2,
                  validation_data=(X_test,Y_test),
                  callbacks=keras_callbacks)

#guardamos el modelo
model.save(common.model_file_name)

#importamos el modelo a javascript
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, './shared/model/')

#o podemos usar el siguiente comando despues de entrenar el modelo

#tensorflowjs_converter \
#  --input_format=keras \
#  /model/model.h5 \
#  /shared/model

train_score=model.evaluate(X_train,Y_train,verbose=2)
valid_score=model.evaluate(X_test,Y_test,verbose=2)

print('Train MAE:',round(train_score[1],4),' Train Loss:',round(train_score[0],4))
print('Val MAE:',round(valid_score[1],4),' Val Loss:',round(valid_score[0],4))

#ver los resultados del entrenamiento
plt.style.use('ggplot')

def plot_history(history,x_size,y_size):
    print(history.keys())
    
    #preparar el dibujo
    plt.rcParams["figure.figsize"]=[x_size,y_size]
    fig,axes=plt.subplots(nrows=4,ncols=4,sharex=True)
    
    #resumir MAE
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'],loc='upper left')
    
    #dibujar los resultados
    plt.draw()
    plt.show()
    
plot_history(history.history,x_size=8,y_size=12)




















