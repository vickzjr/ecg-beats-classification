# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:42:49 2019

@author: vickzjr
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 22:31:14 2019

@author: vickzjr
"""
#title = 'Deep AE 128-64-32 3 class SVDB wt bior epoch 300'
title = 'Deep AE 128-64-32 3 class SVDB wt bior epoch 500'
import numpy as np
from keras.callbacks import ModelCheckpoint
beats = np.loadtxt('F:/Aritmia/new-aritmia/Dataset/SVDB/Beats encoder SVDB non bwr batch 8 {0}.csv'.format(title),delimiter=',')
labels = np.loadtxt('F:/Aritmia/new-aritmia/Dataset/SVDB/labels SVDB non bwr 3 class.csv')


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#beats_scaled = scaler.fit_transform(beats)
from keras.utils import to_categorical

labels = to_categorical(labels)

from sklearn.model_selection import train_test_split
train_data,test_data,train_label, test_label = train_test_split(beats,labels,test_size=0.1,stratify=labels,random_state=42)

model = '{0} dengan data SVDB non bwr batch 8 - DNN 3 HL 3 class AAMI v5'.format(title)
np.savetxt('train data {0}.csv'.format(model),train_data,delimiter=',',fmt='%.3f')
np.savetxt('train label {0}.csv'.format(model),train_label,delimiter=',',fmt='%i')
np.savetxt('test data {0}.csv'.format(model),test_data,delimiter=',',fmt='%.3f')
np.savetxt('test label {0}.csv'.format(model),test_label,delimiter=',',fmt='%i')
from keras.layers import Input,Dense
from keras.models import Model
import random
#random.seed(42)
import time
start_time = time.time()
mc = ModelCheckpoint('best_model {0}.h5'.format(model), monitor='val_acc', mode='max', verbose=1, save_best_only=True)
inputs = Input(shape=(train_data.shape[1],))
x = Dense(100,activation='relu')(inputs)
x = Dense(50,activation='relu')(x)
x = Dense(100,activation='relu')(x)
#x = Dense(50,activation='relu')(x)
#x = Dense(100,activation='relu')(x)
#x = Dense(50,activation='relu')(x)
#x = Dense(100,activation='relu')(x)
outputs = Dense(train_label.shape[1],activation='softmax')(x)
dnn = Model(inputs=inputs,outputs=outputs)
dnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
dnn.fit(train_data,train_label,epochs=100,batch_size=32,validation_data=(test_data,test_label),shuffle=False,callbacks=[mc])
lama = time.time() - start_time

oa = dnn.history.history['acc']
oa = oa[-1]
ova = dnn.history.history['val_acc']
ova = ova[-1]

with open('waktu training model {0}.txt'.format(model), 'w') as f:
    f.write('lama training model {0} {1}'.format(model,lama))
    f.write('Last overall acc {0}'.format(oa))
    f.write('Last overall val acc {0}'.format(ova))

   
dnn.save('{0}.h5'.format(model))

import matplotlib.pyplot as plt
fig, (ax0,ax1)= plt.subplots(nrows=2,figsize=(20,20))
ax0.plot(np.arange(100), dnn.history.history['acc'],'r',label='train')
ax0.plot(np.arange(100),dnn.history.history['val_acc'],'b',label='test')
ax0.set_title('Accuracy')
ax0.set_xlabel('Epochs')
ax0.set_ylabel('Accuracy')
ax0.legend()
ax1.plot(np.arange(100), dnn.history.history['loss'],'r',label='train')
ax1.plot(np.arange(100),dnn.history.history['val_loss'],'b',label='test')
ax1.set_title('Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
plt.savefig('F:/Aritmia/new-aritmia/Khusus/plot loss & accuracy {0}.jpg'.format(model))


import matplotlib.pyplot as plt
fig, (ax0)= plt.subplots(nrows=1,figsize=(5,5))
ax0.plot(np.arange(100), dnn.history.history['loss'],'r',label='train')
ax0.plot(np.arange(100),dnn.history.history['val_loss'],'b',label='test')
ax0.set_title('Loss')
ax0.set_xlabel('Epochs')
ax0.set_ylabel('Loss')
ax0.legend()
plt.savefig('F:/Aritmia/new-aritmia/Khusus/plot loss {0}.jpg'.format(model))
plt.close()

import matplotlib.pyplot as plt
fig, (ax0)= plt.subplots(nrows=1,figsize=(5,5))
ax0.plot(np.arange(100), dnn.history.history['acc'],'r',label='train')
ax0.plot(np.arange(100),dnn.history.history['val_acc'],'b',label='test')
ax0.set_title('Accuracy')
ax0.set_xlabel('Epochs')
ax0.set_ylabel('Accuracy')
ax0.legend()
plt.savefig('F:/Aritmia/new-aritmia/Khusus/plot acc {0}.jpg'.format(model))
plt.close()
