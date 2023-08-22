import os
import queue
import struct

import keyboard
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io.wavfile as wavfile
import scipy.signal
import sounddevice
import soundfile

import path


q = queue.Queue()
stft_list = list()
sample_list = list()
global b
b = False


def get_carrier_freqs(fft:np.ndarray, freq):
    border = fft.mean()+fft.std()
    old_freq = -1
    carrier_freqs = []
    for i in range(0,fft.size-1):
        if fft[i] > border:
            tmp = int(freq[i])
            if tmp != old_freq:
                carrier_freqs.append(int(freq[i]))
                old_freq = int(freq[i])
    old_freq = -1
    start_freq = -1
    freq_range = []
    for freq in carrier_freqs:
        if old_freq == -1:
            start_freq = freq
            old_freq = freq
        else:
            if freq == old_freq+1 or freq == old_freq +2 :
                old_freq = freq
            else:
                freq_range.append((start_freq,old_freq))
                start_freq = freq
                old_freq = freq

    return carrier_freqs, freq_range


def contains(carrierFreqs, frequency):
    return list(filter(lambda x: x[0] <= frequency <= x[1], carrierFreqs))


def callback(indata,frames,time,status):
    #print("Got input ",frames,time)
    q.put(indata.copy())
    tmp = indata.copy().squeeze()
    print(tmp.shape
          )
    sample_list.append(tmp)
    stft_list.append(scipy.signal.stft(tmp))


def record_audio(filename =  os.path.join(path.getpath(),"testRecording.wav")):
    input_device = sounddevice.default.device[0] #device is an array [default Input, default output]
    if filename is None:
        pass
    else:
        with soundfile.SoundFile(filename,mode="w",channels=1,subtype="PCM_32",samplerate=44100) as file:
            with sounddevice.InputStream(samplerate=44100,device=input_device,channels=1,callback=callback):
                print("Start recording")
                try:
                    while True:
                        file.write(q.get())
                        if keyboard.is_pressed("enter"):
                            print("finsihed")
                            break
                except KeyboardInterrupt: # Since Keyboard.is_pressed does not work via SSH
                    print("finished Recording")
    print("Recording stopped")


class audioAnalyzer:
    def __init__(self,filepath = os.path.join(path.getpath(),"test.wav"), use_librosa:bool = False):
        if use_librosa:
            self.samples, self.sample_rate = librosa.load(filepath)
            self.samples = librosa.core.to_mono(self.samples)
        else:
            self.sample_rate, samples = wavfile.read(filepath)
            if samples.shape.__len__() > 1:
                self.samples = samples.sum(axis=1) / 2
            else:
                self.samples = samples

    def show_samples (self, title: str, samples_to_display = None):
        plt.title(title)
        if samples_to_display is not None:
            plt.plot(self.samples[0:samples_to_display])
        else:
            plt.plot(self.samples)
        plt.show()

    def get_frequency(self,samples = None):
        if samples is None:
            samples = self.samples
        fft_librosa = scipy.fft.rfft(samples, norm="ortho")
        freqs = scipy.fft.rfftfreq(int(samples.size), 1./self.sample_rate)
        return fft_librosa, freqs

    def plot_freq(self,title:str,fft, fft_freq):
        plt.plot(fft_freq, fft, "r")
        plt.title(title)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Count dbl-sided')
        plt.show()

    def filt_freq(self,fft,fft_org, line=60):
        filt = []
        i = 0
        for f in fft:

            if (f < line) and (f > 5):
                filt.append(0)
                print(i)
            else:
                filt.append(fft_org[i])
            i += 1

    def irfft(self,fft):
        return scipy.fft.irfft(fft)


##TODO: get amount of time we send the ultrasound (as can be seen in spectogram)
##Get the
    def logscale_spec(self, spec, sr=44100, factor=20.):
        timebins, freqbins = np.shape(spec)

        scale = np.linspace(0, 1, freqbins) ** factor
        scale *= (freqbins - 1) / max(scale)
        scale = np.unique(np.round(scale))

        # create spectrogram with new freq bins
        newspec = np.complex128(np.zeros([timebins, len(scale)]))
        for i in range(0, len(scale)):
            if i == len(scale) - 1:
                tmp = spec[:, int(scale[i]):]
                newspec[:, i] = np.sum(tmp, axis=1)
            else:
                tmp = spec[:, int(scale[i]):int(scale[i + 1])]
                newspec[:, i] = np.sum(tmp , axis=1)

        # list center freq of bins
        allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
        freqs = []
        for i in range(0, len(scale)):
            if i == len(scale) - 1:
                freqs += [np.mean(allfreqs[int(scale[i]):])]
            else:
                freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

        return newspec, freqs

    def get_spec_time(self,specto,t,f,carrierFreqs):

        indexes = []
        for i in range(0,f.size-1):
            tmp = int(f[i])
            if contains(carrierFreqs,f[i]).__len__():
                indexes.append(i)
        threshold = specto.mean()
        xrange = range(0, specto.shape[0])
        yrange = range(0, specto.shape[1])
        values = []
        statlist = []
        for x in xrange:
            for y in yrange:
                if specto[x][y] > threshold:
                    values.append((f[x], t[y],specto[x][y]))
                    statlist.append(specto[x][y])
        line = np.array(statlist).mean()+np.array(statlist).std()
        newValues = []
        setOfFreqs = set()
        for val in values:
            #Sort list to
            if val[2] > line:
                newValues.append(val)
                setOfFreqs.add(val[0])
        setOfFreqs = np.sort(list(setOfFreqs))
        sortedValues = []
        for i in range(setOfFreqs.__len__()-1):
            sortedValues[i] = list()

        for val in newValues:
            index = np.where(np.sort(list(setOfFreqs)) == val[0])[0][0]
            sortedValues[index].append(val)
        print(newValues.__len__())
        oldFreq = 0


    def feature_extraction(self):
        fft,freq = self.get_frequency()
        fft = np.abs(fft)
        carrier_freqs,carrier_freqs_range = get_carrier_freqs(fft,freq)

        f,t,zxx = scipy.signal.stft(self.samples,self.sample_rate,window=scipy.signal.get_window("hann",500),nperseg=500)
        print(zxx.shape)
        plt.title("get_window(\"hann\",512)")
        plt.pcolormesh(t,f,np.abs(zxx))
        plt.colorbar()
        plt.show()


#record_audio()
#audio = audioAnalyzer(os.path.join(path.getpath(),"testRecording.wav"))
#audio.feature_extraction()


#audio = audioAnalyzer()

#audio.show_samples("test")

#audio.get_frequency()
#audio.feature_extraction()
#audio.plot_freq("test",fft,freqs)
