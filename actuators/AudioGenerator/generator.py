import csv
import hashlib
import http.server
import os
import socket
import sys
import time
from threading import Thread

import miio
import miio.exceptions
import numpy as np
import sounddevice
from gtts import gTTS
from pyffmpeg import FFmpeg

from AudioGenerator.audio.audioGenerator import AudioGenerator
from AudioGenerator.audio.textGenerator import TextGenerator
from AudioGenerator.ccrypt import ccrypt


def get_random_number(MAX_VALUE=100):
    return int.from_bytes(os.urandom(32), sys.byteorder) % MAX_VALUE


def convert_audio(mp3file, wavfile):
    ff = FFmpeg()
    ff.convert(mp3file, wavfile)
    print("Converted from mp3 to wav")


def get_file_hash(file):
    return hashlib.md5(file_as_bytes(open(file, "rb"))).hexdigest()


def file_as_bytes(file):
    with file:
        return file.read()


class InstallError(BaseException):
    pass


class Generator:
    def __init__(self, ip="192.168.0.115", token="PLACEHOLDER_HERE"):
        """
        Load or generate/train textGenerator Model
        Connect to MI vac Robot
        IP: "192.168.0.198"
        token: "PLACEHOLDER_HERE" looks like this right now the easiest way to get the token is
        """
        self.ip = ip
        self.token = token
        self.device = miio.vacuum.Vacuum(ip=ip, token=token)
        self.Project_name = os.path.dirname(__file__)
        self.pkg_name = "enc_audio"
        self.pkg_path = os.path.join(self.Project_name, self.pkg_name + ".pkg")
        self.device.home()
        NUM_EPOCHES = 2
        text_file_name = "generated_text_" + str(NUM_EPOCHES) + ".txt"
        data = csv.reader(open(os.path.join(self.Project_name, "audio_en.csv")), delimiter=",")

        text = ""
        for filename, txt in data:
            text += txt
            text += "\n"

        self.textgen = TextGenerator(text, self.Project_name)

        if self.textgen.default_pretrained_exists():
            print("load model")
            self.textgen.load_model()
        else:
            print("train and save model")
            self.textgen.train_model()
            self.textgen.save_model()

    def play_audio(self, retrys=3):
        """
        Simply play the audio file which has been transfered to the mirobot
        """
        try:
            self.device.test_sound_volume()
        except miio.exceptions.DeviceException as e:
            if retrys == 0:
                raise e
            print("device Exception happened")
            self.device = miio.vacuum.Vacuum(ip=self.ip, token=self.token)
            self.play_audio(retrys=retrys - 1)

    def transfer_audio(self):
        fs = FileHoster(self.Project_name)
        fs.start()
        md5 = get_file_hash(os.path.join(self.Project_name, self.pkg_name + ".pkg"))
        url = f"http://{fs.ip}:{fs.port}/{self.pkg_name}" + ".pkg"
        print(url)
        status = self.device.install_sound(url=url, md5sum=md5, sound_id=10000)
        print(status)
        time.sleep(10)
        progress = self.device.sound_install_progress()
        while progress.is_installing:
            if progress.is_errored:
                fs.stop()
                raise InstallError("Error while installing sound")
            print(f"{progress.progress}%")
            time.sleep(2)
            progress = self.device.sound_install_progress()
        print("Installation finished")
        fs.stop()

    def generate_encrypt_audio(self):
        to_speech = ""
        rand_int = get_random_number(10) + 1
        print("random Number = ", rand_int)
        for x in range(rand_int):
            ##Temp op 0.45 or .5 seem to have the best results so not too far away from original but also not to wired
            to_speech = self.textgen.generate_text(temp=0.45)

        print("generated Text: ", to_speech)
        text_speech = gTTS(to_speech)
        print("Google Text to Speech generated")
        path = os.path.join("generated_data")
        os.makedirs(path, exist_ok=True)
        mp3file = os.path.join(self.Project_name, path, "speech.mp3")
        wavfile = os.path.join(self.Project_name, path, "speech.wav")
        text_speech.save(mp3file)

        ff = FFmpeg()
        outFile = ff.convert(mp3file, wavfile)
        print("saved and converted")
        audio_gen = AudioGenerator()
        sample_rate, text_audio = audio_gen.read_audio(outFile)
        print(f"speech sampling rate is {sample_rate}")
        chunk_size = get_random_number(3) + 1  # we want alteast 1 array or a split
        chunks = np.array_split(text_audio, chunk_size)
        mixed_audio = None
        for i in chunks:
            gen_noise, rate, time = audio_gen.generate_audio(rate=sample_rate)
            if mixed_audio is None:
                mixed_audio = np.concatenate((i, gen_noise), axis=None)
            else:
                mixed_audio = np.concatenate((mixed_audio, i, gen_noise), axis=None)
        print("Audio Mixed")
        audio_file_path = audio_gen.write_audio_to_file(self.Project_name, sample_rate, mixed_audio, "start.wav")
        ccrypt.encrypt(self.Project_name, self.pkg_name)


class FileHoster:
    def __init__(self, dir, port=8080):
        self.t = Thread(target=self._run)
        self.ip = socket.gethostbyname_ex(socket.gethostname())[2][- 1]
        self.port = port
        handler = http.server.partial(http.server.SimpleHTTPRequestHandler, directory=dir)
        handler.protocol_version = "HTTP/1.0"
        self.httpd = http.server.ThreadingHTTPServer((self.ip, port), handler)

    def start(self):
        self.t.start()

    def _run(self):
        print("starting server")
        self.httpd.serve_forever()
        print("server closed")

    def stop(self):
        self.httpd.shutdown()
        self.t = Thread(target=self._run)


class Generator_outputdevice:
    def __init__(self, retrain=False):
        self.Project_name = os.path.dirname(__file__)
        NUM_EPOCHES = 2
        self.device = sounddevice.default.device[1]
        self.rate = 44100.0
        text_file_name = "generated_text_" + str(NUM_EPOCHES) + ".txt"
        data = csv.reader(open(os.path.join(self.Project_name, "audio_en.csv")), delimiter=",")

        text = ""
        for filename, txt in data:
            text += txt
            text += "\n"

        self.textgen = TextGenerator(text, self.Project_name)

        if self.textgen.default_pretrained_exists() & (retrain is False):
            print("load model")
            self.textgen.load_model()
        else:
            print("train and save model")
            self.textgen.train_model()
            self.textgen.save_model()

    def play_audio(self):
        """
        Output sound on the selected deivce form sounddevice list
        """
        sounddevice.play(self.audio, samplerate=24000, device=self.device)
        sounddevice.wait()

    def transfer_audio(self):
        """
        No need to do anything here only nesesary for MIIO robot aduio
        :return:
        """
        return
    def generate_encrypt_audio(self):
        to_speech = ""
        rand_int = get_random_number(10) + 1
        print("random Number = ", rand_int)
        for x in range(rand_int):
            ##Temp op 0.45 or .5 seem to have the best results so not too far away from original but also not to wired
            to_speech = self.textgen.generate_text(temp=0.45)

        print("generated Text: ", to_speech)
        text_speech = gTTS(to_speech)
        #text_speech = gTTS("test")
        print("Google Text to Speech generated")
        path = os.path.join("generated_data")
        os.makedirs(path, exist_ok=True)
        mp3file = os.path.join(self.Project_name, path, "speech.mp3")
        wavfile = os.path.join(self.Project_name, path, "speech.wav")
        text_speech.save(mp3file)

        ff = FFmpeg()
        outFile = ff.convert(mp3file, wavfile)
        print("saved and converted")
        audio_gen = AudioGenerator()
        sample_rate, text_audio = audio_gen.read_audio(outFile)
        print(f"speech sampling rate is {sample_rate}")
        chunk_size = get_random_number(3) + 1  # we want alteast 1 array or a split
        chunks = np.array_split(text_audio, chunk_size)
        mixed_audio = None
        for i in chunks:
            gen_noise, rate, time = audio_gen.generate_audio(rate=sample_rate)
            if mixed_audio is None:
                mixed_audio = np.concatenate((i, gen_noise), axis=None)
            else:
                mixed_audio = np.concatenate((mixed_audio, i, gen_noise), axis=None)
        print("Audio Mixed")
        self.audio =  mixed_audio

    def get_output_devices(self):
        devices = sounddevice.query_devices()
        print(devices)
        try:
            self.device = int(input("select device: "))
        except:
            print("faild to get output device default is chosen instead")
            self.device = sounddevice.default.device[1]
        self.rate = sounddevice.query_devices(self.device)["default_samplerate"]

    def print_text(self, number_of_texts, tmp=0.45):
        for x in range(number_of_texts):
            print(self.textgen.generate_text(temp=tmp))


if __name__ == '__main__':
    gen = Generator_outputdevice()
    gen.get_output_devices()
    gen.generate_encrypt_audio()
    gen.play_audio()

