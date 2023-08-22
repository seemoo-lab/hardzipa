import random
import sched
import time
from datetime import timedelta
from threading import Thread

from HueControl.HueLamp.HueClass import HueLamp
from HumidiControl.controller import HumidifierControler
from AudioGenerator.generator import Generator, Generator_outputdevice

HUMID_DEVICE_IP = "192.168.0.100"
HUMID_DEVICE_ID = "PLACEHOLDER_HERE"
HUMID_DEVICE_KEY = "PLACEHOLDER_HERE"

MI_IP = "192.168.0.101"
MI_TOKEN = "PLACEHOLDER_HERE"
BRIDGE_IP = "192.168.0.114"


def print_time(start_time):
    print(timedelta(seconds=time.time() - start_time))


class Controller:
    def __init__(self, audio_source=1):
        """
        Init the actuators
        :param audio_source : 1 if audio should be any connected audio source to this device
                            : 0 if audio should be played via MIIO vacuum clean robot
        """
        if audio_source == 1:
            self.audio = Generator_outputdevice()
            self.audio.get_output_devices()
        else:
            self.audio = Generator(MI_IP, MI_TOKEN)
        self.light = HueLamp()
        self.light_scheduler = None
        self.audio_scheduler = None
        self.humid_scheduler = None
        print("\n**** \n All Actuators Initialized \n **** \n")
        self.thread_list = list()
        self.is_stop = False

    def start(self):
        self.is_stop = False
        t1 = Thread(target=self._audio)
        t2 = Thread(target=self._humid)
        t3 = Thread(target=self._lamp)
        self.thread_list.append(t1)
        self.thread_list.append(t3)
        for thread in self.thread_list:
            thread.start()
        print("Alls Threads Running")

    def stop(self):
        self.is_stop = True
        print("Waiting for all threads to finish")
        for thread in self.thread_list:
            thread.join()
        print("All threads finished")

    def _audio(self):
        self.audio_scheduler = sched.scheduler(time.time, time.sleep)
        start_time = 0
        while True:
            if self.is_stop:
                return
            if self.audio_scheduler.empty():
                if start_time != 0:
                    print("audio took:")
                    print_time(start_time)
                self.audio_scheduler.enter(self._get_random_audio_time(), 1, lambda: self.audio.play_audio())
                self.audio.generate_encrypt_audio()
                self.audio.transfer_audio()
                start_time = time.time()
            self.audio_scheduler.run(False)
            time.sleep(1)

    def _lamp(self):
        self.light_scheduler = sched.scheduler(time.time, time.sleep)
        start_time = 0
        while True:
            if self.is_stop:
                return
            if self.light_scheduler.empty():
                if start_time != 0:
                    print("light took:")
                    print_time(start_time)
                self.light_scheduler.enter(self._get_random_light_time(), 1, lambda: self.light.random_light())
                start_time = time.time()
            self.light_scheduler.run(False)
            time.sleep(1)

    def _humid(self):
        self.humid_scheduler = sched.scheduler(time.time, time.sleep)
        start_time = 0
        while True:
            if self.is_stop:
                return
            if self.humid_scheduler.empty():
                if start_time != 0:
                    print("Humidifier took:")
                    print_time(start_time)
                self.humid_scheduler.enter(self._get_random_humidity_time(), 1, lambda: self.humid.spray_random(True))
                start_time = time.time()
            self.humid_scheduler.run(False)
            time.sleep(1)

    def _get_random_audio_time(self):
        # between 1 and 3 minutes
        return random.randint(30, 90)

    def _get_random_light_time(self):
        # between 30 sec and 1,5 min
        return random.randint(30, int(1.5 * 60))

    def _get_random_humidity_time(self):
        return random.randint(5 * 60, 10 * 60)

    def join(self):
        for t in self.thread_list:
            t.join()
            print("thread finished")


if __name__ == '__main__':
    controller = Controller()
    controller.start()
    controller.join()
