#!/usr/bin/env python
# coding: utf-8

# In[17]:


#using https://github.com/markadivalerio/audio-classifier-project/blob/master as a reference
#using https://medium.com/@anonyomous.ut.grad.student/building-an-audio-classifier-f7c4603aa989 as reference
#using https://towardsdatascience.com/tagged/audio-classification?p=6244954665ab
# https://github.com/d4r3topk/comparing-audio-files-python/blob/master/mfcc.py
#https://github.com/e-alizadeh/medium/blob/master/notebooks/intro_to_dtw.ipynb
import pandas as pd
import numpy as np
import os
import wave
from pydub import AudioSegment
import librosa
import noisereduce as nr
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt


# In[2]:


base_folder = 'WhaleSpeciesData/'
target = 'WhaleSpeciesData/Wave/'
labels = ['Artificial', 'Dolphin', 'Fish', 'Seal', 'Whale']


# In[9]:


#convert files from mp3 to wav
for folder in labels:
    filesInFolder = os.listdir(base_folder + folder)
    for file in filesInFolder:
        destinationFileName = target + folder +'/'+ file.split('.')[0] + '.wav'
        fileToRead = base_folder + folder + '/' + file
        sound = AudioSegment.from_mp3(fileToRead)
        sound.export(destinationFileName, format="wav")


# In[41]:


for folder in labels:
    filesInFolder = os.listdir(target + folder)
    for waveFile in filesInFolder:
        file = target + folder + '/' + waveFile
        wavefile = wave.open(file, 'r')
        bytesequence = wavefile.readframes(-1)
        print(file, len(np.frombuffer(bytesequence, dtype = 'int16')))


# In[43]:


wavefile = wave.open('WhaleSpeciesData/Wave/Artificial/Vessel.wav','r')
bytesequence = wavefile.readframes(-1)
val = np.frombuffer(bytesequence, dtype = 'int16')
val


# In[ ]:


frames = bytesequence.getnframes()
rate = bytesequence.getframerate()
duration = frames / float(rate)


# In[ ]:


wavefile.getframerate()


# In[21]:


def plot_wav(wav_file, label=None):
    librosa_load, librosa_sampling_rate = librosa.load(wav_file)
    scipy_sampling_rate, scipy_load = wav.read(wav_file)
    fig = plt.figure(figsize=(12, 4))
    plt.plot(scipy_load)
    plt.show()
def plotAllWavs():
    for folder in labels:
        filesInFolder = os.listdir(base_folder + folder)
        for file in filesInFolder:
            destinationFileName = target + folder +'/'+ file.split('.')[0] + '.wav'        
            plot_wav(destinationFileName, destinationFileName)


# In[37]:


def createFileToLabelDict():
    fileLabel = dict()
    filePaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(target) for f in filenames if os.path.splitext(f)[1] == '.wav']
    for filePath in filePaths:
        key = filePath
        value = filePath.split('/')[2]
        fileLabel[key] = value
    return fileLabel
    


# In[39]:


fileLabel = createFileToLabelDF()


# In[109]:


def compute_accumulated_cost_matrix(x, y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x, y)

    # Initialization
    cost = np.zeros((len(y), len(x)))
    cost[0,0] = distances[0,0]
    
    for i in range(1, len(y)):
        cost[i, 0] = distances[i, 0] + cost[i-1, 0]  
        
    for j in range(1, len(x)):
        cost[0, j] = distances[0, j] + cost[0, j-1]  

    # Accumulated warp path cost
    for i in range(1, len(y)):
        for j in range(1, len(x)):
            cost[i, j] = min(
                cost[i-1, j],    # insertion
                cost[i, j-1],    # deletion
                cost[i-1, j-1]   # match
            ) + distances[i, j] 
            
    return cost

def compute_euclidean_distance_matrix(x, y) -> np.array:
    """Calculate distance matrix
    This method calcualtes the pairwise Euclidean distance between two sequences.
    The sequences can have different lengths.
    """
    dist = np.zeros((len(y), len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            vect = x[j]-y[i]
            dist[i,j] = np.dot(vect,vect)
    return dist


# In[120]:


import librosa
import librosa.display
import matplotlib.pyplot as plt
#from fastdtw import fastdtw

from scipy.spatial.distance import euclidean
from numpy.linalg import norm

#Loading audio files
y1, sr1 = librosa.load('WhaleSpeciesData/Wave/Whale/ringtoneAtlRightWhale.wav') 
y2, sr2 = librosa.load('WhaleSpeciesData/Wave/Dolphin/BottleNoseDolphin.wav') 

#Showing multiple plots using subplot
plt.subplot(1, 2, 1) 
mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)
from dtw import *
alignment = dtw(mfcc1.T, mfcc2.T, keep_internals=True)
cost = compute_accumulated_cost_matrix(mfcc1.T, mfcc2.T)
print("The normalized distance between the two : ",dtw_distance)   # 0 for similar audios 

plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.plot(path[0], path[1], 'w')   #creating plot for DTW

plt.show()  #To display the plots graphically

#


# In[123]:


import numpy as np

## A noisy sine wave as query
idx = np.linspace(0,6.28,num=100)
query = np.sin(idx) + np.random.uniform(size=100)/10.0

## A cosine is for template; sin and cos are offset by 25 samples
template = np.cos(idx)

## Find the best match with the canonical recursion formula
from dtw import *
alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
alignment.plot(type="threeway")

## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
dtw(query, template, keep_internals=True, 
    step_pattern=rabinerJuangStepPattern(6, "c"))\
    .plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
print(rabinerJuangStepPattern(6,"c"))
rabinerJuangStepPattern(6,"c").plot()


# In[126]:


import librosa
import matplotlib.pyplot as plt
from dtw import dtw

#Loading audio files
y1, sr1 = librosa.load('WhaleSpeciesData/Wave/Whale/ringtoneAtlRightWhale.wav') 
y2, sr2 = librosa.load('WhaleSpeciesData/Wave/Whale/ringtoneAtlRightWhale.wav') 

#Showing multiple plots using subplot
plt.subplot(1, 2, 1) 
mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
librosa.display.specshow(mfcc1)

plt.subplot(1, 2, 2)
mfcc2 = librosa.feature.mfcc(y2, sr2)
librosa.display.specshow(mfcc2)

dist, cost, acc_cost, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print("The normalized distance between the two : ",dist)   # 0 for similar audios 

plt.imshow(cost.T, origin='lower', cmap=plt.get_cmap('gray'), interpolation='nearest')
plt.plot(path[0], path[1], 'w')   #creating plot for DTW

plt.show()  #To display the plots graphically


# In[ ]:




