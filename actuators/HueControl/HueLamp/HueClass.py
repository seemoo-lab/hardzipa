import random
import time

import numpy as np
import phue
import requests
import upnpy

serial_no_bulb = "PLACEHOLDER_HERE"


class NoDeviceFoundException(Exception):
    pass


COLOR_CODING = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}


def convert_type(R):
    """Coverts type to int if its Hex"""
    if type(R) is not int:
        R = int(R, 16)
    return R


def correct_gamma(color):
    if color <= 0.0031308:
        color *= 12.92
    else:
        color = (1.0 + 0.055) * pow(color, (1.0 / 2.4)) - 0.055
    return color


class HueLamp:
    """ Hue Lamp Controller via Hue Bridge (BLE version won't work, they need some encryption)
        For now we expect only one bridge but this could easily be extended
    """

    def __init__(self, Bridge_Ip=None):
        if Bridge_Ip is None:
            Bridge_Ip = self._discover_service()
        self.bridge = phue.Bridge(Bridge_Ip)
        self.bridge.connect()
        self.switch_on_off(False)
        # self.bridge.register_app()

    def add_lamp(self, serial_no: str = serial_no_bulb):
        data = {"deviceid": [serial_no]}
        return self.bridge.request("POST", "/api/" + self.bridge.username + "/lights", data)

    def random_light(self):
        blinking = random.randint(0, 1)
        color = self.get_random_color()
        duration = self._get_duration()
        print(f"Blinking:{bool(blinking)},Color:{color}, Duration:{duration}")
        if blinking:
            duty = self._get_blinking_duty()
            blink_duration = 0
            self.switch_on_off(True)
            self.set_Color(color[0], color[1], color[2])
            b = False
            while blink_duration < duration:
                self.switch_on_off(b)
                time.sleep(duty)
                blink_duration += duty
                b = not b
            self.switch_on_off(False)
        else:
            self.switch_on_off(True)
            self.set_Color(color[0], color[1], color[2])
            time.sleep(duration)
            self.switch_on_off(False)

    def _get_blinking_duty(self):
        # blinking every 1 to 5 seconds i dont want to get an epileptic seizure
        return random.uniform(1, 5)

    def _get_duration(self):
        # from 5 seconds to 2 minutes
        return random.uniform(5, 60)

    def set_Color(self, R, G, B):
        """Set Color in Hex [0x00 - 0xFF] or integer[0-255]"""
        R = convert_type(R)
        G = convert_type(G)
        B = convert_type(B)

        x, y = self.rgb_to_xy(R, G, B)
        for light in self.bridge.lights:
            try:
                light.xy = [x, y]
            except:
                pass

    def switch_on_off(self, b: bool):
        for light in self.bridge.lights:
            try:
                light.on = b
            except:
                pass

    @staticmethod
    def _discover_service():
        my_bridge_id = 'PLACEHOLDER_HERE'
        upnp = upnpy.UPnP()
        dev = upnp.discover(5)

        for d in dev:
            if d.friendly_name.__contains__("Philips hue"):
                return d.host  # UPnP discovery seems to be not always working ...
        res = requests.get("https://discovery.meethue.com/")
        devices = res.json()
        for dev in devices:
            if dev["id"] == my_bridge_id:
                return dev['internalipaddress']
        raise NoDeviceFoundException("UPnP discovery could not find a device in the network")

    @staticmethod
    def rgb_to_xy(r: int, g: int, b: int):
        """
        http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
        Got the transformation matrix from here.
        According to the documentation we should use CIE color space

        r, g, b values should be 0 - 255
        """
        m_CIE_RGB = """ 0.4887180  0.3106803  0.2006017;
                        0.1762044  0.8129847  0.0108109;
                        0.0000000  0.0102048  0.9897952"""
        sRGB = """  0.4124564  0.3575761  0.1804375;
                    0.2126729  0.7151522  0.0721750;
                    0.0193339  0.1191920  0.9503041"""
        git_matrix = """
                    0.649926 0.103455 0.197109;
                    0.234327 0.743075 0.022598;
                    0.0000000 0.053077 1.035763"""
        r = r / 255
        g = g / 255
        b = b / 255
        M = np.asmatrix(m_CIE_RGB)
        RGB = np.array([[r], [g], [b]])
        X, Y, Z = M * RGB
        sum = X + Y + Z
        x = X / sum
        y = Y / sum
        return x.item(), y.item()

    @staticmethod
    def xy_to_RGB(x: int, y: int):
        """https://github.com/PhilipsHue/PhilipsHueSDK-iOS-OSX/blob/00187a3db88dedd640f5ddfa8a474458dff4e1db/ApplicationDesignNotes/RGB%20to%20xy%20Color%20conversion.md"""
        git_mat = """
             1.4628067 -0.1840623 -0.2743606;
            -0.5217933  1.4472381  0.0677227;
            0.0349342 -0.0968930  1.2884099"""
        z = 1.0 - x - y
        Y = 254  # Max brightness range from 1 - 254
        X = (Y / y) * x
        Z = (Y / y) * z
        M = np.asmatrix(git_mat)
        vec = [[X], [Y], [Z]]
        R, G, B = M * vec
        # gamma correction
        R = correct_gamma(R.item())
        G = correct_gamma(G.item())
        B = correct_gamma(B.item())
        return R * 255, G * 255, B * 255

    def get_random_color(self):
        color_index = random.randint(0, 2)
        return COLOR_CODING[color_index]


if __name__ == '__main__':
    l = HueLamp()
    l.add_lamp("PLACEHOLDER_HERE")
    l.add_lamp("PLACEHOLDER_HERE")
