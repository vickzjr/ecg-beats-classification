# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:09:20 2019

@author: vickzjr
"""


import glob
import os
import pywt
import numpy as np


#Take all file with extension you want to glob and return list of directory path of each record in database
#example = 'F:/Aritmia/new-aritmia/Dataset/Incart DB/*.atr'
def glob_items(path):

    paths_data = glob.glob(path)
    path_split = []
    
    for x in range(len(paths_data)):
        valuee = paths_data[x]
        temp = os.path.splitext(valuee)[0]
        
        path_split.append(temp)
        
    return path_split


#checking the label on the record to make 4 super classes of beats classification based on AAMI
def check_label(label):
    if label == 'L' or label == 'R' or label == 'e' or label == 'j':
        label = 'N'
    elif label == 'E':
        label = 'V'
    elif label == 'A' or label == 'a' or label == 'J':
        label = 'S'
    elif label == '/' or label == 'f':
        label = 'Q'
    return label

"""
Denoising the signal using wavelet transform, 
define first about level and wavelet family name

signal is signal reference
level is decomposition level of signal
name is wavelet family name that available in pywt library like (bior6.8, sym5, etc)
wavelet_transform(signal,level,name)

example :
new_signal = wavelet_transform(signal,8,'sym5')
"""
def wavelet_transform(signal,level,name):
    
    a = signal
    w = pywt.Wavelet(name)
    ca = []
    cd = []

    for level in range(level):
        (a, d) = pywt.dwt(a, w)
        ca.append(a)
        cd.append(d)
    a = [0] * len(a)
    a = np.array(a)
    cd.append(a)

    from statsmodels.robust import mad

    sigma = mad( cd[0] )
    uthresh = sigma * np.sqrt( 2*np.log( len(signal)))

    new_cd = []
    for d in cd:
        new_cd.append(pywt.threshold(d, value=uthresh, mode="soft"))

    new_cd.reverse()
    new_signal = pywt.waverec(new_cd, w)
    
    return new_signal