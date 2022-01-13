#!/usr/bin/env python
# coding: utf-8

# In[264]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Conv1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

import wave
import pandas as pd
from pathlib import Path
from scipy.io import wavfile
import numpy as np
from sklearn.model_selection import train_test_split
import os


# In[156]:


class ExportFromWaveFile():
    def __init__(self, lengthInSeconds, writeToFolder):
        self.lengthInSeconds = lengthInSeconds
        self.writeToFolder = writeToFolder
        
    def extractMiddle(self,wavFile):
        plus_minus_seconds = self.lengthInSeconds/2.

        output = wavfile.read(wavFile)
        sample_rate = output[0]
        arr = output[1]
        midpoint_arr = len(arr)/2
        plus_minus_seconds_sample_rate = sample_rate * plus_minus_seconds
        start,stop = int(midpoint_arr - plus_minus_seconds_sample_rate), int(midpoint_arr + plus_minus_seconds_sample_rate)
        return arr[start:stop], sample_rate
    
    def recursiveExtractFromFolders(self, folderPath):
        allFiles = list(Path(folderPath).rglob("*.[wW][aA][vV]"))
        folder = folderPath.split('/')[-2]
        filename = folderPath.split('/')[-1] +'.wav'
        for filename in allFiles:
            extractedArr, sample_rate = self.extractMiddle(filename)
            new_folder = self.writeToFolder + '/' + folder
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            wavefile.write(filename, sample_rate, extractedArr)  
    
        
        newAudio.export('newSong.wav', format="wav")
            
            


export = ExportFromWaveFile(5, '/WhaleSpeciesData/Wave/8000KHZ')
output_arr = export.recursiveExtractFromFolders('/Users/gabeshindnes/Downloads/animal_data_for_nn_models/boat')




# In[239]:


class BuildDataFromAudioFile():
    def __init__(self, folderToReadFilesFrom):
        self.folderToReadFilesFrom = folderToReadFilesFrom
        self.data = pd.DataFrame()
        self.data_arr = []
        self.label = []
        self.label_to_int = {}
        self.int_to_label = {}
        
    def readFiles(self):
        allFiles = list(Path(self.folderToReadFilesFrom).rglob("*.[wW][aA][vV]"))
        for filename in allFiles:
            output = wavfile.read(filename)
            if(len(output[1]) == 40000):
                arr = output[1]
                if(str(filename).split('/')[-2] == 'Artificial copy'):
                    continue
                elif(type(arr) is np.ndarray):
                    if(arr.size == 80000):
                        row_data = arr[:,0]
                        row_data = (row_data - row_data.mean())/row_data.std()
                        self.data_arr.append(row_data)
                        self.label.append(str(filename).split('/')[-2])
                    elif(arr.size == 40000):
                        arr = (arr - arr.mean())/arr.std()
                        self.data_arr.append(arr)
                        self.label.append(str(filename).split('/')[-2])
            elif(len(output[1]) > 40000):
                sample_rate = output[0]
                arr = output[1]
                midpoint_arr = len(arr)/2
                start,stop = int(midpoint_arr - 20000), int(midpoint_arr + 20000)
                arr = output[1][start:stop]
                arr = (arr - arr.mean())/arr.std()
                if(len(arr.shape) == 1):
                    self.data_arr.append(arr)
                    label = str(filename).split('/')[-2]
                    self.label.append(label)
                else:
                    self.data_arr.append(np.asarray(arr[:,0]))
                    label = str(filename).split('/')[-2]
                    self.label.append(label)
        self.data = pd.DataFrame(self.data_arr)
        self.data['label'] = self.label
        return self.data
    
    
    def changeToBinaryLabel(self):
        dataFrame = self.data
        dataFrame['label'] = self.label
        dataFrame['label'] = np.where(dataFrame['label'] == 'boat', 1, np.where(dataFrame['label'] == 'Artificial', 1, np.where(dataFrame['label'] == 'background', 1, 0)))
        self.label_to_int = {}
        self.label_to_int['artificial'] = 1
        self.label_to_int['animal'] = 0
        self.int_to_label = {}
        self.int_to_label[1] = 'artificial'
        self.int_to_label[0] = 'animal'
        return dataFrame
    
    def createLabelToIntDictionaries(self):
        label_col = set(self.data['label'])
        for i,y in enumerate(label_col):
            self.label_to_int[y] = i
            self.int_to_label[i] = y
        self.data['int_label'] = self.data['label']
        self.data['int_label'] = self.data['int_label'].map(self.label_to_int)
    


# In[265]:


class BuildModel():
    def __init__(self, input_dim, output_dim, activation_function):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_function = activation_function
        self.model = Sequential()
        
    def CreateModel(self):
        self.model.add(Dense(10000, input_dim = self.input_dim, activation = self.activation_function))
        self.model.add(Dense(5000,  activation = self.activation_function))
        self.model.add(Dense(1000,  activation = self.activation_function))
        self.model.add(Dense(256,  activation = self.activation_function))
        self.model.add(Dense(32,  activation = self.activation_function))
        self.model.add(Dense(16,  activation = self.activation_function))
        self.model.add(Dense(self.output_dim , activation= self.activation_function))
        self.model.compile(optimizer='sgd', loss=CategoricalCrossentropy(), metrics = ['accuracy'])
    
    def TrainModel(self, xtrain, ytrain, epochs):
        self.model.fit(xtrain,ytrain, epochs = epochs)
        
    def PredictModel(self, xtest, int_to_label):
        ymodeltest = self.model.predict(xtest)
        classes = np.argmax(ymodeltest, axis = 1)
        return [*map(int_to_label.get,classes)]
     
def SplitData(X,y, to_cat = True):
    x_train, x_test, y_train, y_test = train_test_split(X,y)
    if(to_cat):
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    return x_train,x_test,y_train,y_test


# In[266]:


buildData = BuildDataFromAudioFile('./WhaleSpeciesData/Wave/8000KHZ')
data = buildData.readFiles()
buildData.createLabelToIntDictionaries()
              
data = buildData.changeToBinaryLabel()




# In[267]:


xtr, xtst, ytr, ytst = SplitData(data.drop(['label', 'int_label'],axis = 1), data['label'], True)
model = BuildModel(40000, len(buildData.int_to_label), 'sigmoid')
model.CreateModel()
model.TrainModel(xtr, ytr,20)


# In[270]:


#predictions = model.PredictModel(xtst, buildData.int_to_label)
#print(predictions)
#for i in model.model.predict(xtst):
#    print(np.argmax(i))
#buildData.int_to_label
#for i in ytst:
#    print(i)

model.model.save('animal_or_artificial')

