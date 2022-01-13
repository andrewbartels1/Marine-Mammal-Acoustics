#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Nov 11 21:11:58 2021
@author: bartelsaa
"""
# boiler plate stuff
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

# settings.configure()
# import django.conf as conf
# conf.settings.DATABASES['default']['ENGINE'] = 'django.db.backends.sqlite3'
# settings.configure()
from django import db
print(db.connections.databases)
print("DB NAME ")


from pams.models.audio_clip import AudioClip

# Features that need to be dervied
# id | lat | lon | pam_sensor_name | start_time | end_time | audio_filepath | low_res_spec_filepath | bounding_boxes

data_schema = {"sensor_platform":         str,
               "label":                   str,
               "min_y":                   int,
               "min_x":                   int,
               "max_y":                   int,
               "max_x":                   int,
               "min_f":                 float,
               "min_t":                 float,
               "max_f":                 float,
               "max_t":                 float,
               "centroid_y":            float,
               "centroid_x":            float,
               "duration_x":            float,
               "bandwidth_y":           float,
               "area_xy":                 int,
               "centroid_f":            float,
               "centroid_t":            float,
               "duration_t":            float,
               "bandwidth_f":           float,
               "area_tf":               float,
               "audio":                   str,
               "start_time":    'datetime64[ns]',
               "end_time":      'datetime64[ns]'}


# not implemented, for troubleshooting/visualization, run the make spec function and collect the variables running around in there
def display_results(Sxx_power, X5, Sxx_db_noNoise_smooth_finale, df_centroid, freq_sample_in_hz, ext, df_rois, display=True):
    # if display is true
    if display:
        # Visualize
        Sxx_db = power2dB(Sxx_power)
        plot2d(power2dB(Sxx_power), title="raw")
        plot2d(X5, title="Smoothed as itll get")
        plot2d(Sxx_db_noNoise_smooth_finale,
               title="Smoothed as itll get Final")
        ax0, fig0 = overlay_rois(
            create_mask(
                im=Sxx_db_noNoise_smooth_finale,
                mode_bin="relative",
                bin_std=8,
                bin_per=0.5,
                verbose=True,
                display=display,
            ),
            df_rois,
        )
        ax1, fig = plot2d(Sxx_db)
        ax1, fig = overlay_rois(dB2power(Sxx_power), df_centroid)
        ax1.scatter(df_centroid.centroid_x, df_centroid.centroid_y)
        ax1.set_title("centroid overlays")
        fig
        for i in range(len(df_centroid)):
            write(
                "audio_snippet_test_{}.wav".format(i),
                freq_sample_in_hz,
                df_centroid.audio_clip[i],
            )

        # for viewing spec chips
        for i in range(len(df_centroid)):
            ax, fig = plot2d(
                df_centroid["spec_cropped"][i], **{
                    "extent": [
                        df_centroid["tn_crop"][i].min(),
                        df_centroid["tn_crop"][i].max(),
                        ext[2],
                        ext[3],
                    ],
                    "title":
                    "centroid {i}, freq={freq}, time={time}".format(
                        i=i,
                        freq=df_centroid.centroid_f[i],
                        time=df_centroid.centroid_t[i],
                    ),
                })
            ax.scatter(df_centroid.centroid_t[i], df_centroid.centroid_f[i])
            fig.savefig(
                "centroid_{i}, freq_{freq}, time_{time}.png".format(
                    i=i,
                    freq=df_centroid.centroid_f[i],
                    time=df_centroid.centroid_t[i]),
                dpi=1000,
            )

# this does all the algorithm stuff
def make_spec(wav_file,
              display=False,
              min_roi_pixel_area_to_omit=2500,
              max_roi_pixel_area_to_omit=None,
              time_around_feature=5,
              verbose=False):

    audio_signal, freq_sample_in_hz = load(
        wav_file, display=display)  # replace with wav_file
    
    # downsample to what everythgin else is at
    # audio_signal =  resample(audio_signal, freq_sample_in_hz, _get_sample_rate(wav_file)) 
    # Make spectrogram
    Sxx_power, time_vector, frequency_vector, ext = spectrogram(
        audio_signal,  _get_sample_rate(wav_file))

    # take out some noise
    Sxx_power_noNoise = median_equalizer(Sxx_power)

    # First we remove the stationary background in order to increase the contrast [1]
    # Then we convert the spectrogram into dB
    Sxx_db_noNoise = power2dB(Sxx_power_noNoise)
    Sxx_db_noNoise_smooth = smooth(Sxx_db_noNoise, std=0.25)

    # do some removing along the axis and smooth to get better shapes
    X4, noise_profile4 = remove_background_along_axis(Sxx_db_noNoise_smooth,
                                                      axis=1,
                                                      N=50)
    X5, noise_profile5 = remove_background_along_axis(X4,
                                                      mode="median",
                                                      axis=0)
    Sxx_db_noNoise_smooth_finale = smooth(
        X5, std=2)  # heavy handed smoothing on large areas are ok for now

    # create regions of interest and downsample to ones that are of a reasonable size (configurable)
    mask = create_mask(
            im=Sxx_db_noNoise_smooth_finale,
            mode_bin="relative",
            bin_std=8,
            bin_per=0.5,
            verbose=False,
            display=display)
    im_rois, df_rois = select_rois(mask,
                                    display=display,
                                    min_roi=4000,
                                    max_roi=6000,
                                    verbose=verbose)

    # format, use this for cropping
    df_rois = format_features(df_rois, time_vector, frequency_vector)

    # get the final dataframe of features and format them properly
    df_centroid = centroid_features(Sxx_db_noNoise_smooth_finale, df_rois,
                                    im_rois)
    df_centroid = format_features(df_centroid, time_vector, frequency_vector)

    # Filter for some attributes
    df_centroid = df_centroid[(df_centroid.centroid_t > 1) &
                              (df_centroid.centroid_t < time_vector.max() - 1)]

    df_centroid = df_centroid.sort_values(by=["area_tf", "centroid_t"],
                                          ignore_index=True,
                                          ascending=(False, False))
    
    # crop the spec around the centroid position
    if len(df_centroid) > 1:
        df_centroid["spec_cropped"] = df_centroid.apply(
            lambda row: _crop_spec(
                row,
                Sxx_db_noNoise_smooth_finale,
                time_vector,
                frequency_vector,
                time_around_feature,
            ),
            axis=1,
        )

        # separate spec and time array for chip
        df_centroid[['spec_cropped','tn_crop']] = pd.DataFrame(df_centroid.spec_cropped.values.tolist(), index= df_centroid.index)
        # set audio array
        df_centroid["audio_clip"] = df_centroid.apply(
            lambda row: _crop_audio(row, audio_signal, _get_sample_rate(wav_file)), axis=1)
        
        # get start and end time of chip
        df_centroid['chip_start_seconds'] = pd.to_timedelta(df_centroid.apply( lambda row: _get_start_min_time_crop(row), axis=1), 's')
        df_centroid['chip_end_seconds'] = pd.to_timedelta(df_centroid.apply( lambda row: _get_end_max_time_crop(row), axis=1), 's')
        
        df_centroid['audio'] = 'placeholder'
        
        df_centroid = df_centroid.drop(['tn_crop', 'spec_cropped','labelID'], axis=1)
    else:
        df_centroid = None
    return df_centroid


def flac_to_wav(flac_file, remove_flac=False):
    wav_file = os.path.splitext(flac_file)[0] + ".wav"
    # flac = AudioSegment.from_file(flac_file, format='flac')
    # stream = io.BytesIO()
    # flac.export(stream, format='wav')
    os.system("ffmpeg -nostats -loglevel panic -hide_banner -i {inputfile} {output} -y".format(inputfile=flac_file,
                                                       output=wav_file))
    if remove_flac:
        os.remove("{flac_file}".format(flac_file=flac_file))
        print("removing {}".format(flac_file))

    return wav_file

def wipe_table(AudioClip):
    AudioClip.objects.all().delete()
    print("WIPING TABLE!")

# Using the Maad package to load and clean spec
def _crop_spec(row, Sxx_db_noNoise_smooth_finale, time_vector, frequency_vector,
               time_around_feature):
    crop_end = row.centroid_t + time_around_feature
    crop_start = row.centroid_t - time_around_feature

    if crop_start < time_vector.min():
        crop_start = time_vector.min()
    if crop_end > time_vector.max():
        crop_end = time_vector.max()

    S_xx_crop, tn_new, _ = crop_image(
        Sxx_db_noNoise_smooth_finale,
        time_vector,
        frequency_vector,
        fcrop=(frequency_vector.min(), frequency_vector.max()),
        tcrop=(crop_start, crop_end),
    )
    return (S_xx_crop, tn_new)


def _crop_audio(row, audio_signal, freq_sample_in_hz):
    return trim(audio_signal,
                freq_sample_in_hz,
                min_t=row["tn_crop"].min(),
                max_t=row["tn_crop"].max())

def _get_start_min_time_crop(row):
    return row.tn_crop.min()

def _get_end_max_time_crop(row):
    return row.tn_crop.max()

def _get_sample_rate(soundfile):
    # need to downsmaple!!
    sample_freqs = {"adeon": 8000, "SanctSound": 48000}

    if "AMAR" in soundfile:
        sample_freq = sample_freqs["adeon"]
    elif "Sant" in soundfile:
        sample_freq = sample_freqs["SanctSound"]

    return sample_freq

def get_meta_data(wav_file):
    return glob(os.path.join(os.path.dirname(os.path.dirname(wav_file)), 'metadata', "*.json"))

def _parse_path(path):
    """
    formats:
        AMAR504.4.20180501T210951Z.wav
        SanctSound_GR03_02_671666216_190502113002.flac
    """
    time_str = re.split("[._]", path)[-2]
    format = "%y%m%d%H%M%S" if "_" in path else "%Y%m%dT%H%M%SZ"

    return datetime.strptime(time_str, format)

def _make_chip_id_path(temp_last_row, output_folder, row_id):
    # Make dir and set to audio_path column
    temp_last_row.audio_path = os.path.join(output_folder,  str(row_id), "")
    clip_path = pathlib.Path(temp_last_row.audio_path)
    clip_path.mkdir(parents=True, exist_ok=True)
    return clip_path 

def _to_django(row, AudioClip, output_folder, wav_file, AlwaysDownSampleTo=8000):
    
    # convert to dict
    row_dict = row.to_dict()
    raw_audio_file = row_dict.pop("audio_clip")
    

    # could probably do this with a wild card AudioClip(**row_dict) but I want readable
    audio = AudioClip(label=row_dict['label'],
                      min_y=row_dict['min_y'],
                      min_x=row_dict['min_x'],
                      max_y=row_dict['max_y'],
                      max_x=row_dict['max_x'],
                      min_f=row_dict['min_f'],
                      min_t=row_dict['min_t'],
                      max_f=row_dict['max_f'],
                      max_t=row_dict['max_t'],
                      centroid_y=row_dict['centroid_y'],
                      centroid_x=row_dict['centroid_x'],
                      duration_x=row_dict['duration_x'],
                      bandwidth_y=row_dict['bandwidth_y'],
                      area_xy=row_dict['area_xy'],
                      centroid_f=row_dict['centroid_f'],
                      centroid_t=row_dict['centroid_t'],
                      duration_t=row_dict['duration_t'],
                      bandwidth_f=row_dict['bandwidth_f'],
                      area_tf=row_dict['area_tf'],
                      audio_path=row_dict['audio'],
                      chip_start_seconds=make_aware(row_dict['start_time']), # make_aware() gives it the django UTC setting time zone
                      chip_end_seconds=make_aware(row_dict['end_time']),
                      sensor_platform=row_dict['sensor_platform'],
                      lat=row_dict['lat'],
                      long=row_dict['long'])
    
    # save the record and get the id
    audio.save()
    row_id = audio.id
    
    # get the row back to update audio path
    temp_last_row =  AudioClip.objects.get(id=row_id)
    
    output_path = _make_chip_id_path(temp_last_row, output_folder, row_id)
    
    
    print("SAVING AUDIO TO PATH {} AS {}".format(output_path, os.path.join(temp_last_row.audio_path, 'chip.wav')))
    temp_last_row.save()
    
        
    # ALWASY DOWNSAMPLE TO 8000 Hz BEFORE SENDING
    # raw_audio_file =  resample(raw_audio_file, freq_sample_in_hz, AlwaysDownSampleTo)
    
    # actually write chip to files
    print(len(raw_audio_file))
    
    # actually write chip to files
    write(os.path.join(temp_last_row.audio_path, 'chip.wav'), AlwaysDownSampleTo, raw_audio_file) # "chip.wav"
    file_size = os.path.getsize(os.path.join(temp_last_row.audio_path, 'chip.wav'))
    print("File Size is :", file_size, "bytes")
    

def main():
    pd.set_option('display.max_columns', None)
    
    # deployments! (select as many as you want to run at once!)
    data_folder = "data/jax*/"
    # data_folder = "data/hat*/"
    # data_folder = "data/chb*/"
    # data_folder = "data/ble*/"
    # data_folder = "data/sav*/"
    # data_folder = "data/vac*/"
    # data_folder = "data/wil*/"
    
    # defaults
    max_rows_to_send = 2
    lossless_file_type = "**/*.flac"
    output_folder = "data/processed_data/"
    
    # make the audio list with glob
    raw_audio_list = glob(data_folder + lossless_file_type, recursive=True)
    
    print("LEN OF RAW AUDIO FILES FOUND {}".format(len(raw_audio_list)))
    
    # loop through all the audio.flac files
    # make this a function in a class
    for idx, i in enumerate(sorted(raw_audio_list, reverse=True)):
        print("processing {} of {}".format(idx, len(raw_audio_list)))
              
        sensor = Sensor(get_meta_data(i)[0])
        
        wav_file = flac_to_wav(i, remove_flac=True) #make this a stream no leaving a bunch of wav files around
        
        df_centroid =  make_spec(wav_file,
                      display=True,
                      min_roi_pixel_area_to_omit=3900, # NOTE: these are in the area^2 of the image (not ideal)
                      max_roi_pixel_area_to_omit=None,
                      time_around_feature=5)
        
        
        # Calc start time  
        if df_centroid is not None:       
            print(wav_file)
            print("len df_cent {}".format(len(df_centroid)))
            
            start_time = _parse_path(os.path.split( wav_file)[-1])
            
            df_centroid['start_time'], df_centroid['end_time'] = [start_time + df_centroid['chip_start_seconds'],start_time + df_centroid['chip_end_seconds']]
            
                    
            df_centroid['sensor_platform'], df_centroid['lat'], df_centroid['long'] = [sensor.name, sensor.lat, sensor.long]
            
            # clean up
            df_centroid = df_centroid.drop(['chip_start_seconds', 'chip_end_seconds'], axis=1)
            df_centroid = df_centroid.astype(data_schema)
            
            # limit values
            df_centroid = df_centroid.sort_values('area_xy',ascending = False).head(max_rows_to_send)
            
            # the audio_raw column needs to be saved as preprocessed_data/id/chip.wav 
            df_centroid.apply(lambda row: _to_django(row, AudioClip, output_folder, wav_file), axis=1)
        print("removing {}".format(wav_file))
        os.remove(wav_file)
        
if __name__ == "__main__":
    main()

#  From https://www.ncei.noaa.gov/maps/passive_acoustic_data/
# Start Date: 2018-01-09
# End Date: 2021-12-31
# Try to remove background noise
#  https://scikit-maad.github.io/_auto_examples/2_advanced/plot_remove_background.html#sphx-glr-auto-examples-2-advanced-plot-remove-background-py
# Maybe preprocess or label with some unsupervised jazz
# https://scikit-maad.github.io/_auto_examples/2_advanced/plot_unsupervised_sound_classification.html#sphx-glr-auto-examples-2-advanced-plot-unsupervised-sound-classification-py