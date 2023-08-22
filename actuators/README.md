# Requirements

- Raspberry Pis connected to the same network (we used Raspberry Pi 3 Model B).
- At least Python 3.x with
    - Tensorflow
    - phue
    - upnpy
    - pytuya
    - pyffmpeg
    - sounddevice
    - gtts

# Configure

- In startService.py
    - Set the ssh password for the Raspberry Pis.
    - Set the Raspberry pis IPs in the list of IPs.
- In Actuators/Controller.py
    - For Humidifier
        - Set HUMID_DEVICE_IP, HUMID_DEVICE_ID, HUMID_DEVICE_KEY (for Tuya api Humidifier).
    - For MiRobot
        - MI_IP, MI_TOKEN (There are several tutorials on howto extract the token + IP).
    - For the Light bulbs
        - If no IP is provided the Bridge is being discovered.
- Raspberry Pis need the PiRecorder directory in their Home folder.
- Raspberry Pis need a connected microphone (via USB), a connected spg30 gas sensor (via pins), and connected Samsung Galaxy smartphone (via USB) with the RGBreader app.
- Launch the RGBReader app on the phones and accept the USB connections for the devices after the "start" command has been executed.

# Run

- Run python StartService.py
    - Press shift+enter to get the input prompt.
    - Enter "start" to launch the script on all devices.
    - Wait a few seconds to ensure all devices are up.
    - Enter "run" to start recording with all devices.
    - Enter "actuator" to start the actuator algorithm.
        - Default time is 1 h and 30 minutes (you might want to extend this to enter dynamic times).
