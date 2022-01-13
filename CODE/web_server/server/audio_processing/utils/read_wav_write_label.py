#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 23:07:14 2021

@author: bartelsaa
"""
import re, os, io
from glob import glob
import pandas as pd
import sys
import pytz
# from sqlite3 import Error
from models import Sensor
from datetime import datetime, timezone
import pathlib
from maad.rois import (select_rois, create_mask)
from maad.features import (centroid_features)
from maad.sound import (load, resample, spectrogram, remove_background, median_equalizer,
                        remove_background_morpho, remove_background_along_axis,
                        sharpness, spectral_snr, trim, write, smooth)
from maad.util import (power2dB, plot2d, dB2power,format_features, overlay_rois,
                       overlay_centroid, crop_image)

# setup django settings and configs
import django

# Setup django env and add to models
sys.path.append("/app")
os.environ['DJANGO_SETTINGS_MODULE'] = 'pams.settings'
django.setup()

from django.utils.timezone import make_aware

from django import db
print(db.connections.databases)
print("DB NAME ")


from pams.models.audio_clip import AudioClip


output_folder = "data/processed_data/"
pd.set_option('display.max_columns', None)
print("STARTING INSERT")
from audio_processing.utils import things_found
import sqlite3
# from pams.
boat_hits = things_found.boat
critter_hits = things_found.critter
background = things_found.background

def __temp_insert_labels_to_db():


    boat_hits_keys    = list(map(lambda e: (e, "boat"), boat_hits))    
    critter_hits_keys = list(map(lambda e: (e, "critter"), critter_hits))    
    background_keys   = list(map(lambda e: (e, "background"), background))
    # output_folder = "data/processed_data/"
    for idd,e in boat_hits_keys:
        print('updating key {} to value {}'.format(idd, e))
        
        temp_last_row =  AudioClip.objects.get(id=idd)
        temp_last_row.label = e
        temp_output_folder = os.path.join(output_folder,  str(e), "")
        # print("temp_output_folder {}".format(temp_output_folder))
        clip_path = pathlib.Path(os.path.join(temp_output_folder, str(idd), ""))
        # print(clip_path)
        # print(os.path.join(temp_output_folder, str(idd), ""))
        clip_path.mkdir(parents=True, exist_ok=True)
        print(clip_path)
        chip_file, fs = load(os.path.join(temp_last_row.audio_path, 'chip.wav'))
        # print(os.path.join(clip_path, 'chip.wav'))
        write(os.path.join(clip_path, 'chip.wav'), 8000, chip_file) # "chip.wav"
        
    for idd,e in critter_hits_keys:
        print('updating key {} to value {}'.format(idd, e))
        
        temp_last_row =  AudioClip.objects.get(id=idd)
        temp_last_row.label = e
        temp_output_folder = os.path.join(output_folder,  str(e), "")
        print("temp_output_folder {}".format(temp_output_folder))
        clip_path = pathlib.Path(os.path.join(temp_output_folder, str(idd)))
        clip_path.mkdir(parents=True, exist_ok=True)
        chip_file, fs = load(os.path.join(temp_last_row.audio_path, 'chip.wav'))
        print(chip_file)
        write(os.path.join(clip_path, 'chip.wav'), 8000, chip_file) # "chip.wav"

    for idd,e in background_keys:
        print('updating key {} to value {}'.format(idd, e))
        
        temp_last_row =  AudioClip.objects.get(id=idd)
        temp_last_row.label = e
        temp_output_folder = os.path.join(output_folder,  str(e), "")
        print("temp_output_folder {}".format(temp_output_folder))
        clip_path = pathlib.Path(os.path.join(temp_output_folder, str(idd)))
        clip_path.mkdir(parents=True, exist_ok=True)
        chip_file, fs = load(os.path.join(temp_last_row.audio_path, 'chip.wav'))
        print(chip_file)
        write(os.path.join(clip_path, 'chip.wav'), 8000, chip_file) # "chip.wav"
        
    return None

if __name__ == "__main__":
    __temp_insert_labels_to_db()
    