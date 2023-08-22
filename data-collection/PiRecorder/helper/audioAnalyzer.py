import os
import sounddevice
import soundfile
import queue
from threading import Thread
import keyboard
import datetime


class AudioRecorder:
    def __init__(self,sub_dir = None, filename = "test.wav"):
        if sub_dir is None:
            self.filename =os.path.join(os.getcwd(), "testRecording.wav")
            self.audio_time = os.path.join(os.getcwd(), "audio.time")
        else:
            self.filename =os.path.join(os.getcwd(),sub_dir ,"testRecording.wav")
            self.audio_time =os.path.join(os.getcwd(),sub_dir ,"audio.time")
        self.run = True
        print("Audio file locations s",filename)
        self.t = Thread()

    def _callback(indata, frames, time, status):
        pass
        #q.put(indata.copy())

        # Maybe do other stuff as well

    def _start_record_thread(self,input_device):
        q = queue.Queue()
        with soundfile.SoundFile(self.filename, mode="w", channels=1, subtype="PCM_16", samplerate=44100) as file:
            with sounddevice.InputStream(samplerate=44100, device=input_device, channels=1, callback=lambda indata,frames, time, status: q.put(indata.copy())):
                with open(self.audio_time,"w") as audio_file:
                    start_time = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S.%f")
                    audio_file.write(start_time)
                    print(f"Start recording at {start_time}")
                try:
                    while self.run:
                        file.write(q.get())
                        if keyboard.is_pressed("enter"):
                            print("finsihed")
                            break
                except KeyboardInterrupt:  # Since Keyboard.is_pressed does not work via SSH
                    print("finished via keyboard interrupt")
        print("Recording stopped")

    def record_audio(self):
        input_device = sounddevice.default.device[0]  # device is an array [default Input, default output]
        if self.filename is None:
            pass
        else:
            self.t = Thread(target=self._start_record_thread,args=(input_device,))
            self.t.start()

    def stop_recording(self):
        self.run = False
        self.t.join()
        print("audio Stopped")
