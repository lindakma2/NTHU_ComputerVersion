# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 07:59:29 2023

@author: a3311
"""


from fastai import * 
from fastai.vision import *
import pathlib
import os
from scipy import signal
from scipy.io import wavfile
from fastprogress import progress_bar
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

audio_type='road' #choose one group name
name="belinda" #your name to seperate the photo

for i in range (10):
    audio_data = 'audio/'+audio_type+' ('+str(i+1)+').mp3'
    x , sr = librosa.load(audio_data)
    print(x.shape, sr)#(220207,) 22050
    librosa.load(audio_data, sr=44100)
    ipd.Audio(audio_data)
    
    ###############################################
    
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    save_data='seg_train/'+audio_type+'/TD_'+audio_type+'_'+name+str(i*5+1)+'.png'
    plt.savefig(save_data)
    X = librosa.stft(x) #stft轉換的function，可以用參數調整
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log') #photo2 經stft轉換後的圖
    save_data='seg_train/'+audio_type+'/FD_'+audio_type+'_'+name+str(i*5+1)+'.png'
    plt.savefig(save_data)
    
    '''
    data argumentation
    ===================================
    加上噪音
    Noise addition using normal distribution with mean = 0 and std =1
    Permissible noise factor value = x > 0.004
    '''
    file_path = audio_data
    wav, sr = librosa.load(file_path,sr=None)
    wav_n = wav + 0.009*np.random.normal(0,1,len(wav))
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(wav_n, sr=sr)
    save_data='seg_train/'+audio_type+'/TD_'+audio_type+'_'+name+str(i*5+2)+'.png'
    plt.savefig(save_data)
    Wav_n = librosa.stft(wav_n) #stft轉換的function，可以用參數調整
    Wav_ndb = librosa.amplitude_to_db(abs(Wav_n))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Wav_ndb, sr=sr, x_axis='time', y_axis='log') #photo2 經stft轉換後的圖
    save_data='seg_train/'+audio_type+'/FD_'+audio_type+'_'+name+str(i*5+2)+'.png'
    plt.savefig(save_data)
    '''
    Shifting the sound wave
    Permissible factor values = sr/10
    '''
    wav_roll = np.roll(wav,int(sr/10))
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(wav_roll, sr=sr)
    save_data='seg_train/'+audio_type+'/TD_'+audio_type+'_'+name+str(i*5+3)+'.png'
    plt.savefig(save_data)
    Wav_roll = librosa.stft(wav_roll) #stft轉換的function，可以用參數調整
    Wav_rolldb = librosa.amplitude_to_db(abs(Wav_roll))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Wav_rolldb, sr=sr, x_axis='time', y_axis='log') #photo2 經stft轉換後的圖
    save_data='seg_train/'+audio_type+'/FD_'+audio_type+'_'+name+str(i*5+3)+'.png'
    plt.savefig(save_data)
    
    
    '''
    Time-stretching the wave
    Permissible factor values = 0 < x < 1.0
    '''
    
    factor = 0.4
    wav_time_stch = librosa.effects.time_stretch(wav,rate=factor)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(wav_time_stch, sr=sr)
    save_data='seg_train/'+audio_type+'/TD_'+audio_type+'_'+name+str(i*5+4)+'.png'
    plt.savefig(save_data)
    Wav_time_stch = librosa.stft(wav_time_stch) #stft轉換的function，可以用參數調整
    Wav_time_stchdb = librosa.amplitude_to_db(abs(Wav_time_stch))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Wav_time_stchdb, sr=sr, x_axis='time', y_axis='log') #photo2 經stft轉換後的圖
    save_data='seg_train/'+audio_type+'/FD_'+audio_type+'_'+name+str(i*5+4)+'.png'
    plt.savefig(save_data)
    '''
    pitch shifting of wav
    Permissible factor values = -5 <= x <= 5
    '''
    wav_pitch_sf = librosa.effects.pitch_shift(wav,sr=sr,n_steps=-5)
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(wav_pitch_sf, sr=sr)
    save_data='seg_train/'+audio_type+'/TD_'+audio_type+'_'+name+str(i*5+5)+'.png'
    plt.savefig(save_data)
    Wav_pitch_sf= librosa.stft(wav_pitch_sf) #stft轉換的function，可以用參數調整
    Wav_pitch_sfdb = librosa.amplitude_to_db(abs(Wav_pitch_sf))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Wav_pitch_sfdb, sr=sr, x_axis='time', y_axis='log') #photo2 經stft轉換後的圖
    save_data='seg_train/'+audio_type+'/FD_'+audio_type+'_'+name+str(i*5+5)+'.png'
    plt.savefig(save_data)



