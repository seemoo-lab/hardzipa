import os
import socket
import sys
import time
from datetime import timedelta
from threading import Thread

import keyboard
import ssh2.exceptions
from ssh2.session import Session

from Acutuators import Controller

pi_pass = "PLACEHOLDER_HERE"
download_chunk_size = 2 ** 15  # Max size to read at once

ips = ["192.168.0.104", "192.168.0.105", "192.168.0.106", "192.168.0.109", "192.168.0.110"]
files = ["gas.txt", "RGB.txt", "testRecording.wav"]
src = "/home/pi/"
baseDir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "Experiment")

should_execute = False


class SCPException(BaseException):
    pass


def kill_sessions(ip: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, 22))

    session = Session()
    session.handshake(sock)
    session.userauth_password("pi", pi_pass)

    channel = session.open_session()
    channel.execute("tmux kill-session")
    print(f"Killed Tmux session on device {ip}")
    channel.close()


def _connect_and_execute(ip: str = "192.168.0.101", args: str = "-H 1 -m 30"):
    command = f"tmux new-session -d \; send-keys \"sudo python3 PiRecorder/Recorder.py {args} \" Enter"
    # command = f"tmux new-session -d \; send-keys \"sudo echo {args} \" Enter"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, 22))

    session = Session()
    session.handshake(sock)
    session.userauth_password("pi", pi_pass)

    channel = session.open_session()
    channel.execute(command)
    print(f"Executed on device {ip} \n \t with command: \t {command}")
    global should_execute
    while not should_execute:
        time.sleep(0.1)
    channel = session.open_session()
    channel.execute("tmux send-keys -t 0 Enter")
    print(f"should execute executed on {ip}")
    channel.close()


def start_all(args: str = "-H 1 -m 30", wait_to_finishe=True):
    connection_threads = list()
    if args == "copy":
        print("Enter Folder name to save data")
        foldername = input()
    for address in ips:
        if args == "copy":
            deviceID = address.split(".").pop()
            t = Thread(target=scp_get,
                       kwargs=dict(ip=address, src=src, dest=os.path.join(baseDir, foldername, deviceID)))
            print(f"Started scp for device {address}")
            # t.join()
            print("finished")
        elif args == "kill":
            t = Thread(target=kill_sessions, kwargs=dict(ip=address))
        else:
            t = Thread(target=_connect_and_execute, kwargs=dict(ip=address, args=args))
        connection_threads.append(t)
        t.start()
    if wait_to_finishe:
        for thread in connection_threads:
            thread.join()
        print("Executed on all Hosts")
    else:
        print("not waiting to finsih")
        return connection_threads


def scp_get(ip, src, dest, retry=2):
    if retry == 0:
        raise SCPException(f"Failed to download from host {ip}")
    os.makedirs(dest, exist_ok=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, 22))

    session = Session()
    session.handshake(sock)
    session.userauth_password("pi", pi_pass)
    for f in files:
        print(f"Device {ip} Downloading {f}")
        try:
            session_get(session, os.path.join(src, f), os.path.join(dest, f))
        except ssh2.exceptions.SSH2Error as e:
            print(e)
            sock.close()
            scp_get(ip, src, dest, retry - 1)
        print(f"{ip}: Destination: " + os.path.join(dest, f))
    print(f"Host with ip : {ip} is done")


def session_get(session, src, dst):
    chan, info = session.scp_recv2(src)
    final_size = info.st_size
    current_size = 0
    print(f"Downloading File of size: {info.st_size}")
    with open(dst, "wb") as localFile:
        start_time = time.time()
        while True:
            size, b = chan.read(download_chunk_size)  # Maybe read the whole file ?
            localFile.write(b)
            current_size += size
            if current_size < final_size:
                sys.stdout.write(f"\r\t {current_size}/{final_size} \t")
            if current_size >= final_size:
                sys.stdout.flush()
                sys.stdout.write(f"Download Complete in {timedelta(seconds=time.time() - start_time)} \t")
                chan.close()
                return True


def print_commands():
    print(""" !!!!!!! The actuator will not work properly if the cmd is stuck at input, therfore shift+space enables input !!!!!!!!!! \n
    start \t - Starts all clients [-H <number>] for hour [-m <number>] for minute default: 1 H 30 m and waits for the run command to record 
    run \t - Executes the Recording on all devices
    actuator \t - Starts the actuator algo
    stop \t - Stops the actuator algo
    kill  \t - Kills all running sessions on the host
    status \t - Should show the runtime of the actuator algo
    exit \t - Exit
    quit \t - Exit
    copy \t - Copys the generated data (Needs to be reworked and should not be used)
    help  \t - shows this page
    """)


def main():
    AC_controller = None
    start_time = None
    thread_list = list()
    while True:
        print("press shift+space to enter a Command: ")
        keyboard.wait(
            "shift+space")  ## We will get a problem with Popen when we constantly try to read in, hence we will make a hotkey to read in data
        sys.stdout.write("Enter a command: ")
        command = input().split(" ")

        if command[0].lower() == "actuator":
            minute = 60  # seconds
            hour = 60 * minute
            duration = 1 * hour + 30 * minute
            interval = range(int(duration / 30))
            AC_controller = Controller.Controller(1)
            AC_controller.start()
            overall = 0
            for x in interval:
                overall += 30
                print(timedelta(seconds=overall))
                time.sleep(30)
            print("time is over")
            AC_controller.stop()

        elif command[0].lower() == "start":
            # AC_controller = Controller.Controller()
            # AC_controller.start()
            thread_list = start_all(args="-H 0 -m 30", wait_to_finishe=False)
        elif command[0].lower() == "run":
            global should_execute
            should_execute = True
            start_time = time.time()
            for t in thread_list:
                t.join()
            print("Run is done")
            should_execute = False
        elif command[0].lower() == "stop":
            if not AC_controller is None:
                print("Stopping Actuatros")
                AC_controller.stop()
                AC_controller = None
                print("Actuators stopped")
        elif command[0].lower() == "status":
            if start_time is None or AC_controller is None:
                print("Nothing has been started or is currently running")
            else:
                print(timedelta(seconds=time.time() - start_time))
        elif command[0].lower() == "kill":
            start_all("kill")
        elif command[0].lower() == "copy":
            start_all("copy")
        elif command[0].lower() == "exit" or command[0].lower() == "quit":
            print("Quit Python ")
            exit(0)
        else:
            print_commands()
        time.sleep(2)


if __name__ == '__main__':
    main()
