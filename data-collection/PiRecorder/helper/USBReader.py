from helper.AOA.AndroidOpenAccessory import Accessory
import os


class USBReader:
    def __init__(self,sub_dir=None):
        self.AOA = Accessory()
        self.AOA.add_read_callback(self.readCallback)
        if sub_dir is None:
            self.file = open(os.path.join(os.getcwd(), "RGB.txt"), "w")
        else:
            self.file = open(os.path.join(os.getcwd(),sub_dir ,"RGB.txt"), "w")
        self.counter = 0

    def readCallback(self,str):
        self.file.write(str)

        if self.counter > 50: #only print every 20th recive
            print(str)
            self.counter = 0
        self.counter += 1

    def __del__(self):
        self.file.close()

    def start(self):
        self.AOA.write("start".encode())

    def stop(self):
        self.AOA.write("start".encode())
        self.AOA.close()