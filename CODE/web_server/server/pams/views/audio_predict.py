import random
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pams.models.classification import Classification
from pams.serializers import ClassificationSerializer
from tensorflow.keras.models import load_model
import librosa
from scipy.io import wavfile
import maad
import os
import numpy as np


@api_view(['POST'])
def process_audio(request):
	species = ['Animal', 'Artificial']

	file = request.FILES.get('audio', None)
	filename = file.name if file else None


	output = wavfile.read(file)
	maad_resample = maad.sound.resample(output[1], output[0], 8000)
	resample_numpy = np.asarray(maad_resample)
	if(len(resample_numpy.shape) > 1):
		resample_numpy = resample_numpy[:,0]

	model = load_model('/app/pams/animal_or_artificial_model.h5')
	midpoint = len(resample_numpy)/2
	start,stop = int(midpoint - 20000), int(midpoint + 20000)
	arr = np.asarray(resample_numpy[start:stop])
	mfcc_output = librosa.feature.mfcc(arr.astype(np.float), sr = 8000, n_mfcc = 40)
	flatten_mfcc = np.asarray([item for sublist in mfcc_output for item in sublist])
	flatten_mfcc = flatten_mfcc.reshape(1, len(flatten_mfcc))
	serializer = ClassificationSerializer(Classification(species[np.argmax(model.predict(flatten_mfcc))], filename))

	return Response(serializer.data, status=status.HTTP_200_OK)
