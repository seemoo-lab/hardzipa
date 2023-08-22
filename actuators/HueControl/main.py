import os
import time

from HueLamp.HueClass import HueLamp

print(__file__)
path = os.path.dirname(__file__)

h = HueLamp()

h.set_Color(255,0,0)
time.sleep(10)
h.set_Color(0,255,0)
time.sleep(10)
h.set_Color(0,0,255)
time.sleep(10)
h.switch_on_off(False)