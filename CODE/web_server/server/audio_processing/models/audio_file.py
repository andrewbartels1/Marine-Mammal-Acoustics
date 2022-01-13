from datetime import datetime
import os
import re


class AudioFile:
    def __init__(self, flac_path):
        self.datetime = self._parse_path(flac_path)
        self.wav_filename = self._flac_to_wav(flac_path)

    def _flac_to_wav(flac_file, remove_flac=False):
        wav_file = os.path.splitext(flac_file)[0] + ".wav"
        os.system(f"ffmpeg -i {flac_file} {wav_file}")

        if remove_flac:
            os.system("rm {flac_file}".format(flac_file=flac_file))
            print("removing {}".format(flac_file))

        return wav_file

    def _parse_path(self, path):
        """
        formats:
            AMAR504.4.20180501T210951Z.wav
            SanctSound_GR03_02_671666216_190502113002.flac
        """
        time_str = re.split("[._]", path)[-2]
        format = "%y%m%d%H%M%S" if "_" in path else "%Y%m%dT%H%M%SZ"

        return datetime.strptime(time_str, format)
