import random
import time
from threading import Thread

from pytuya import Device

DEVICE_IP = "192.168.0.192"
DEVICE_ID = "PLACEHOLDER_HERE"
DEVICE_KEY = "PLACEHOLDER_HERE"


class HumidifierControler(Device):
    def __init__(self, ip:str, id:str,key:str,dev_type= "device"):
        super(HumidifierControler,self).__init__(id,ip,key,dev_type)
        self.set_version(3.3)

    def on_off_switch(self,switch:bool = False):
        if switch:val = 'True'
        else: val = 'False'
        print(val)
        payload = self.generate_payload("set",{'1':switch})
        data = self._send_receive(payload)

    def intensity(self,strength:int):
        val = "off"
        if strength == 1:
            val = "small"
        if strength == 2:
            val = 'big'
        payload = self.generate_payload("set",{'103':val})
        self._send_receive(payload)

    def light_on_off(self,switch:bool):
        payload = self.generate_payload("set", {'11':switch})
        self._send_receive(payload)

    def spray_random(self,blocking:bool = True):
        duration = random.randint(10*60,15*60)
        intensity = random.randint(1,2)
        if blocking:
            self.spray_for_interval_blocking(intensity,duration)
        else:
            self.spray_for_interval_non_blocking(intensity,duration)

    def spray_for_interval_blocking(self,intensity:int,duration,trys = 5):
        try:
            self.intensity(intensity)
            print("started spraying")
            time.sleep(duration)
            self.intensity(0)
            print("spray finished")
        except BaseException as e:
            if (trys == 0):
                raise e
            self.spray_for_interval_blocking(intensity,duration,trys-1)

    def spray_for_interval_non_blocking(self,intensity:int,duration):
        t = Thread(target=self.spray_for_interval_blocking,kwargs=dict(intensity=intensity,duration=duration))
        t.start()