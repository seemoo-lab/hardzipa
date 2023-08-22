# Description
This code is used to collect audio, illuminance, RGB, CO2, and TVOC data, as described in [our paper](https://arxiv.org/abs/2306.04458). 

In brief, our data collection setup contains a Raspberry Pi Model 3 B with an attached Samson Go microphone (via USB), smartphone Samsung Galaxy S6 (via USB), and an SGP30 multigas sensor (via Raspberry Pi’s pins). 

**PiRecorder** – contains the main functionality to obtain data from the microphone, gas, and smartphone sensors (the code is invoked from PiRecorder/Recorder.py). 

**RGBReader** – contains an Android app that needs to be installed on the Samsung Galaxy S6 to collect illuminance and RGB data.
