import os
import random
import subprocess
import sys
import time
from threading import Thread

import keyboard
import numpy as np
import sounddevice as sd
from scipy import signal as sg
from scipy.io import wavfile


class AudioGenerator:
    def __init__(self):
        self.generated_sound_to_play = None
        self.start_idx = 0
        self.exit_is = False
        self.live_amp=1.0
        self.dev = 15

    def generate_audio(self,
                       rate=44100,  # Sample rate in Hz", default=44100
                       freq=19100,  # Frequency in Hz, 19100 ultrasound freq
                       time_length=0.5,  # duration in seconds
                       channels=2,  # Number of channels to produce
                       bits=16,  # Number of bits in each sample
                       ):
        freq = self.get_random_frequency()
        amp = self.get_random_amp()
        time_length = self.get_random_time()
        print(f"Generated soundwave with freq = {freq}, amp = {amp} and {time_length} seconds")
        t = np.linspace(0, time_length, rate * time_length)
        return self._wave_sin(freq,t,amp=amp), rate,time_length

    def generate_and_save(self,rate=44100,freq=800,time_length=10):
        t = np.linspace(0, time_length, rate * time_length)
        self.generated_sound_to_play = self._wave_sin(freq,t,amp=1), rate,time_length

    def run_new_terminal(self):
        command=f"python {os.path.abspath(__file__)}"
        process = subprocess.Popen(command,creationflags=subprocess.CREATE_NEW_CONSOLE,stdin=subprocess.PIPE,stdout=subprocess.PIPE,shell=True)

    def play_sound(self,rate=44100,amp = 0.5,freq=150):
        samplerate = sd.query_devices(self.dev, 'output')['default_samplerate']
        try:
            samplerate = sd.query_devices(self.dev, 'output')['default_samplerate']

            def callback(outdata, frames, time, status):
                if status:
                    print(status, file=sys.stderr)
                t = (self.start_idx + np.arange(frames)) / samplerate
                t = t.reshape(-1, 1)
                outdata[:] = self.live_amp * (0.2*self._wave_saw(freq,t))
                self.start_idx += frames

            with sd.OutputStream(device=self.dev, channels=1, callback=callback,
                                 samplerate=samplerate):
                time.sleep(5400)
        except KeyboardInterrupt:
            return
        except Exception as e:
            print(e)
            return

    def generate_audio_stair(self,
                       rate=44100,  # Sample rate in Hz", default=44100
                       freq = 19100,  # Frequency in Hz, 19100 ultrasound freq
                       time_length = 10,  # duration in seconds
                       channels = 2,  # Number of channels to produce
                       bits = 16,  # Number of bits in each sample
                       amp = 0.5,  # Amplitude of the wave on a scale of 0.0-1.0.
                       ):
        third_octave_band = [40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]
        time_length = round(time_length / len(third_octave_band))
        t = np.linspace(0, time_length, rate * time_length)
        audio = None
        for freq in third_octave_band: #range(200, 20000, 200):
            if audio is None:
                audio = self._wave_sin(freq,t)
            else:
                audio = np.append(audio, self._wave_sin(freq,t))
        dir_path = os.path.dirname(__file__)
        audio = np.asarray(audio, dtype=np.int16)
        file = open(os.path.join(dir_path, "stair.wav"), "wb")
        wavfile.write(file, rate, audio)

    @staticmethod
    def get_random_frequency(Max_freq=22100):
        input = int.from_bytes(os.urandom(32),sys.byteorder)
        return int(input % Max_freq)

    @staticmethod
    def get_random_time(Max_length_sec = 10):
        input = random.uniform(0.5,Max_length_sec)
        return input

    @staticmethod
    def get_random_amp(Max_amp_multiplicator = 1):
        return random.uniform(0,Max_amp_multiplicator)

    @staticmethod
    def read_audio(file):
        return wavfile.read(file)

    @staticmethod
    def write_audio_to_file(path,rate,audio,audio_name):
        file_name = os.path.join(path,audio_name)
        file = open(file_name,"wb")
        wavfile.write(file,rate,audio)
        return file_name

    @staticmethod
    def _wave_sin(freq, t,amp=0.5):
        amp *= np.iinfo(np.int16).max
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)

    @staticmethod
    def _wave_square(freq, t):
        return 0.5 * sg.square(2 * np.pi * freq * t)

    @staticmethod
    def _wave_square_duty(freq, t, duty=0.8):
        return 0.5 * sg.square(2 * np.pi * freq * t, duty=duty)

    @staticmethod
    def _wave_saw(freq, t):
        return 0.5 * sg.sawtooth(2 * np.pi * freq * t)


if __name__ == '__main__':
    gen = AudioGenerator()
    gen.generate_audio_stair(time_length=1800,amp=0.5)
