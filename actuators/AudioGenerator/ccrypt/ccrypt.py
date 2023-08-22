import os
import platform
import subprocess

key = "PLACEHOLDER_HERE"
windows = "cd {} && wsl  tar zc *.wav ^| ccrypt -e -K 'PLACEHOLDER_HERE' > {}.pkg"
linux = "cd {} && tar zc *.wav | ccrypt -e -K 'PLACEHOLDER_HERE' > {}.pkg"


def encrypt(path:str,output:str):
    if platform.system() is "Windows":
        process = subprocess.Popen(windows.format(path,output),stdout=subprocess.PIPE,shell=True)
        process.wait()
        print(process.stdout.read().decode())
    else:
        process = subprocess.Popen(linux.format(path, output), stdout=subprocess.PIPE,shell=True)
        process.wait()
        print(process.stdout.read().decode())


def decrypt(self, data:str):
    pass