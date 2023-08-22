from helper.SensorReader import SensorReader
from helper.audioAnalyzer import AudioRecorder
import time,datetime
import getopt
import sys
import os
from helper.USBReader import USBReader


def print_help():
    print("""
    -h \t Prints Help page \n
    -H \t Hours of recording \n
    -m \t Minutes of recording \n
    -c \t calibrate the sensor for 12 hours \n
    -w \t warm ups the sensor for 10 minutes \n
    -f \t first time wait 1 minute to accept devices
    """)


def main():

    opts,args =getopt.getopt(sys.argv[1:],'fcwhH:m:')
    warmup = False
    calib = False

    first_time_wait = False
    run_hours = 1
    run_min = 30
    for o, arg in opts:
        if o in "-h":
            print_help()
        if o in "-H":
            if arg.isdigit():
                run_hours = int(arg)
        if o in "-m":
            if arg.isdigit():
                run_min = int(arg)
        if o in "-w":
            warmup = True
        if o in "-c":
            calib = True
        if o in "-f":
            first_time_wait = True

    if calib:
        sensor = SensorReader()
        print("Calibrating for 12 hours")
        sensor.calibrate_12h()
        return

    sub_dir = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    sub_path = os.path.join(os.getcwd(),sub_dir)
    os.makedirs(sub_dir,exist_ok=True)
    audio = AudioRecorder(sub_dir=sub_dir)
    sensor = SensorReader(sub_dir=sub_dir)
    light_reader = USBReader(sub_dir=sub_dir)

    if first_time_wait :
        print("accept now all devices")
        time.sleep(60)

    minute = 60
    hour = 60 * minute

    # compute runtime in seconds and div by 30 since we notify every 30 sec how long we are recording
    run_time = int(((hour*run_hours) + (minute*run_min))/30)

    if warmup:
        print("warming up")
        sensor.warm_up(600) #warmup in seconds
        print("Warmup finished")
    print("Press a key to start")
    input()
    print("start recording")
    audio.record_audio()
    sensor.start()
    light_reader.start()
    timesecs = 0
    for x in range(run_time):
        time.sleep(30)
        timesecs+=30
        print("%d:%d passed x = %d" % ((int(timesecs/60)),timesecs%60,x))

    audio.stop_recording()
    sensor.stop()
    light_reader.stop()
    print("stopped")
    return


if __name__ == "__main__":
    main()
