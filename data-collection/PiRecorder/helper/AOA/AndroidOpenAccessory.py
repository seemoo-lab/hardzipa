import usb.core
import usb.util
import fcntl
import struct
import time
import threading
import os
import sys
import socket
import inspect

ACCESSORY_VID = 0x04e8
ACCESSORY_VID_2 = 0x18D1
ACCESSORY_PID = (0x2D00, 0x2D01, 0x2D04, 0x2D05)

MANUFACTURER = "Seemoo"
MODEL_NAME = "RGBReader"
DESCRIPTION = "Reads RGB values from sensor"
VERSION = "0.1"
URL = "http://my-app-is-local"
SERIAL_NUMBER = "1337"

NETLINK_KOBJECT_UEVENT = 15


def main():
    Accessory()


class Accessory:
    def __init__(self):
        self.is_done=False
        self.accessory_task()
        self.callbacks = list()
        self.read_thread = threading.Thread(target=self.read)
        self.read_thread.start()

    def add_read_callback(self,func):
        self.callbacks.append(func)

    def accessory_task(self):
        dev = usb.core.find(idVendor=ACCESSORY_VID);

        if dev is None:
            dev = usb.core.find(idVendor=ACCESSORY_VID_2)
            if dev is None:
                raise ValueError("No compatible device not found")

        print("compatible device found")

        if dev.idProduct in ACCESSORY_PID:
            print("device is in accessory mode")
        else:
            print("device is not in accessory mode yet")

            self.accessory(dev)

            dev = usb.core.find(idVendor=ACCESSORY_VID)

            if dev is None:
                dev = usb.core.find(idVendor=ACCESSORY_VID_2)
                if dev is None:
                    raise ValueError("No compatible device not found")
            print(f"dev kernel is attached ?= {dev.is_kernel_driver_active(0)}")
            if dev.idProduct in ACCESSORY_PID:
                print("device is in accessory mode")

        tries = 1
        while True:
            try:
                if tries <= 0:
                    break
                """dev.reset()
                if dev.is_kernel_driver_active(0) == True:"""
                dev.detach_kernel_driver(0)
                dev.set_configuration()
                break
            except BaseException as e:
                print(e)
                print("unable to set configuration, retrying")
                tries -= 1
                time.sleep(1)

        # even if the Android device is already in accessory mode
        # setting the configuration will result in the
        # UsbManager starting an "accessory connected" intent
        # and hence a small delay is required before communication
        # works properly
        time.sleep(1)
        dev = usb.core.find(idVendor=ACCESSORY_VID)
        if dev is None:
            dev = usb.core.find(idVendor=ACCESSORY_VID_2)
            if dev is None:
                raise ValueError("No device found")
        print("found Configured Device")
        cfg = dev.get_active_configuration()

        if_num = cfg[(0, 0)].bInterfaceNumber
        intf = usb.util.find_descriptor(cfg, bInterfaceNumber=if_num)
        self.ep_out = usb.util.find_descriptor(
            intf,
            custom_match= \
                lambda e: \
                    usb.util.endpoint_direction(e.bEndpointAddress) == \
                    usb.util.ENDPOINT_OUT
        )
        print("found out Descriptor")
        self.ep_in = usb.util.find_descriptor(
            intf,
            custom_match= \
                lambda e: \
                    usb.util.endpoint_direction(e.bEndpointAddress) == \
                    usb.util.ENDPOINT_IN
        )
        print("found In descriptor")
        print("Done")
        length = -1

    def read(self):
        while True:
            if self.is_done:
                print("USB read thread is done")
                self.callbacks.clear()
                return
            try:
                data = self.ep_in.read(1024,timeout=0)
                str = data.tobytes().decode()
                for func in self.callbacks:
                    func(str)
            except usb.core.USBError as e:
                print("failed to send IN transfer")
                print(e)
                break

    def write(self,data:bytearray or bytes):
        try:
            length = self.ep_out.write(data, timeout=0)
            print("%d bytes written" % length)
        except usb.core.USBError as e:
            print("error in writer thread")
            print(e)

    def writer(self,ep_out):
        print("writer thread started")
        while True:
            try:
                length = ep_out.write(b"Hi", timeout=0)
                print("%d bytes written" % length)
                time.sleep(0.5)
            except usb.core.USBError:
                print("error in writer thread")
                break

    def close(self):
        self.is_done=True
        self.read_thread.join()
        self.read_thread = None
        self.callbacks.clear()

    def accessory(self,dev):
        version = dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_IN,
            51, 0, 0, 2)

        print("version is: %d" % struct.unpack('<H', version))

        assert dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            52, 0, 0, MANUFACTURER) == len(MANUFACTURER)

        assert dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            52, 0, 1, MODEL_NAME) == len(MODEL_NAME)

        assert dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            52, 0, 2, DESCRIPTION) == len(DESCRIPTION)

        assert dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            52, 0, 3, VERSION) == len(VERSION)

        assert dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            52, 0, 4, URL) == len(URL)

        assert dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            52, 0, 5, SERIAL_NUMBER) == len(SERIAL_NUMBER)

        dev.ctrl_transfer(
            usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_OUT,
            53, 0, 0, None)

        time.sleep(2)


if __name__ == "__main__":
    main()
