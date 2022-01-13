#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 16:39:21 2021

@author: bartelsaa
"""

# def save_wav_file(raw_audio, folder="data/spec_samples/", normalize=False, plot=False):
#     for idx, flac in enumerate(raw_audio): #164157
#         flac = matching[0]
#         # print(flac)
#         filename_wav = folder + "".join(flac.split('/')[-1]).rsplit('.', 1)[0] + '.wav'

#         sample_freq = get_sample_rate(flac)
#         sig, fs = librosa.load(flac, sr=sample_freq)
#         sf.write(filename_wav, sig, fs)

#         if plot:
#             from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#             filename_png = "data/spec_samples/" + "".join(flac.split('/')[-1]).rsplit('.', 1)[0] + '.png'

#             # print("data/spec_samples/" + "".join(flac.split('/')[-1]) + ".png")
#             print("pushing out spec # {}".format(1))
#             f = plt.Figure()
#             f.set_figheight(15)
#             f.set_figwidth(35)
#             canvas = FigureCanvas(f)
#             ax = f.add_subplot(111)

#             if normalize:
#                 sig = librosa.util.normalize(sig)

#             D = librosa.amplitude_to_db(librosa.stft(np.abs(sig)), ref=np.max)  # STFT of y

#             img = librosa.display.specshow(np.abs(D), ax=ax, y_axis='linear', x_axis='time')
#             ax.set(title="".join(flac.split('/')[-1]))
#             f.colorbar(img, ax=ax, format="%+2.f dB")
#             plt.show()
#             f.savefig(filename_png,bbox_inches='tight')

#             plt.close(f)
#             pyplot.close()
#         del D, sig, fs
