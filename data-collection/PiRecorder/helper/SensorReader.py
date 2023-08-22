import time
import board
import busio
import adafruit_sgp30
from gpiozero import DigitalInputDevice
from threading import Thread
import os
from datetime import datetime


def getTime():
    return datetime.now().isoformat(timespec="milliseconds")+"\t"


class SensorReader:
    def __init__(self,DOPPLER_GPIO = None,sub_dir= None):
        #PIR_pin = 5 #PIR sensor on D5
        if DOPPLER_GPIO is None:
            self.record_motion = False
            self.dp = None
            self.MotionFile = None
        else:
            self.record_motion = True
            self.DOPPLER_GPIO = DOPPLER_GPIO #Dopplar radar on 5v Vin gnd to gnd and out to GPIO17 = PIN11
            self.dp = DigitalInputDevice(17, pull_up=True)
            self.MotionFile = open(os.path.join(os.getcwd(), "motion.txt"), "w")
        self.delay = 0.01
        self.t = Thread(target= self.__start_thread_recording)
        self.tccs = Thread(target=self.__recordCCS)
        self.stop_flag = False
        i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
        if sub_dir is None:
            self.gasFile = open(os.path.join(os.getcwd(), "gas.txt"),"w")
        else:
            self.gasFile = open(os.path.join(os.getcwd(),sub_dir, "gas.txt"),"w")

        self.sgp30 = adafruit_sgp30.Adafruit_SGP30(i2c)
        self.sgp30.iaq_init()
        #self.__setBaseline()

    def __del__(self):
        self.stop_flag = True
        try:
            self.t.join()
        except RuntimeError:
            pass
        self.gasFile.close()
        if self.MotionFile is not None:
            self.MotionFile.close()

    def warm_up(self, seconds:int):
        if seconds <= 0 : return
        print("warming up Sensors for %d:%d to get accurate values" % ((int(seconds/60)),seconds%60))
        print("Calibrating")
        t = Thread(target=self.__print_readings)
        self.stop_flag = False
        t.start()
        time.sleep(seconds)
        self.write_baseline()
        self.stop_flag = True
        t.join()
        print("WarmUp Done")

    def __print_readings(self):
        try:
            while not self.stop_flag:
                print("eCO2 = %d ppm \t TVOC = %d ppb" % (self.sgp30.eCO2, self.sgp30.TVOC))
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("interuppted")
        print("finished readings")
    def calibrate_12h(self):
        """
        Should calibrate the Sensor and calculate the baseline as mentioned here : https://learn.adafruit.com/adafruit-sgp30-gas-tvoc-eco2-mox-sensor/circuitpython-wiring-test
        :return: /
        """
        print("Calibrating")
        t = Thread(target=self.__print_readings)
        self.stop_flag = False
        t.start()
        for x in range(12):
            time.sleep(60*60)
            self.write_baseline()
        self.stop_flag = True
        t.join()
        print("Finished Calibrating")

    def read_baseline(self):
        co2File = os.path.join(os.path.dirname(__file__), "co2base")
        tvocFile = os.path.join(os.path.dirname(__file__), "tvocbase")
        co2Base = None
        tvocBase = None
        try:
            with open(co2File, "r") as file:
                co2Base = int(file.read())
                print(co2Base)
            with open(tvocFile, "r") as file:
                tvocBase = int(file.read())
                print(tvocBase)
        except:
            print("failed to read baseline")
        return co2Base,tvocBase

    def write_baseline(self):
        co2File = os.path.join(os.path.dirname(__file__),"co2base")
        tvocFile = os.path.join(os.path.dirname(__file__),"tvocbase")
        with open(co2File,"w") as file:
            file.write(str(self.sgp30.baseline_eCO2))
        with open(tvocFile,"w") as file:
            file.write(str(self.sgp30.baseline_TVOC))
        print("baseline: CO2",self.sgp30.baseline_eCO2, " TVOC: ",self.sgp30.baseline_TVOC )

    def __start_thread_recording(self):
        print("Sensor Thread started")
        counter = 1
        elapsed_sec = 0
        try:
            while not self.stop_flag:
                counter += 1

                if self.record_motion and self.dp.value:
                    self.MotionFile.write(getTime()+"Motion Detected \n")

                if (counter * self.delay == 1):
                    counter = 1
                    self.gasFile.write(getTime()+"eCO2 = %d ppm \t TVOC = %d ppb" % (self.sgp30.eCO2, self.sgp30.TVOC))
                    elapsed_sec += 1
                    if elapsed_sec > 10:
                        print("eCO2 = %d ppm \t TVOC = %d ppb" % (self.sgp30.eCO2, self.sgp30.TVOC))
                        elapsed_sec = 0
                        self.write_baseline()
                    self.gasFile.write("\n")
                time.sleep(self.delay)
        except KeyboardInterrupt:
            print("Interrupt")
            self.stop()
        print("Sensor Thread Stopped")

    def __recordCCS(self):

        print("warming up Sensor")
        while not self.ccs811.data_ready:
            pass
        print("data ready start Recording")
        try:
            while not self.stop_flag:
                print("CO2: %d ppm, TVOC: %d ppb, %d" % (self.ccs811.eco2, self.ccs811.tvoc,self.ccs811.temperature))
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted")
            self.stop()
    def start(self):
        self.stop_flag = False
        if self.sgp30 is not None:
            self.t.start()
        """if self.ccs811 is not None:
            self.tccs.start()"""
        print("Sensor recoding started")

    def stop(self):
        self.stop_flag = True
        if self.sgp30 is not None:
            self.t.join()
        """if self.ccs811 is not None:
            self.tccs.join()"""
        print("Stopped Successfully")

    def __setBaseline(self):
        co2base, tvocbase = self.read_baseline()
        if co2base is None or tvocbase is None:
            print("No baseLine, Run for 12 Hours to determine Baseline")
            #self.sgp30.set_iaq_baseline(0x8973, 0x8AAE)
        else:
            print("init with previous baseline")
            self.sgp30.set_iaq_baseline(co2base, tvocbase)


if __name__ == '__main__':
    SensorReader
