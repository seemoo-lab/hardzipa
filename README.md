# HardZiPA Codebase
This repository contains the codebase for the HardZiPA system published as: "Hardening and Speeding Up Zero-interaction Pairing and Authentication" by Mikhail Fomichev, Timm Lippert, and Matthias Hollick in *Proceedings of the 2023 ACM International Conference on Embedded Wireless Systems and Networks (EWSN' 23)*.

The relevant datasets can be found on [Zenodo](https://zenodo.org/record/8263497). The pre-print version of our paper is available on [arXiv](https://arxiv.org/abs/2306.04458).

The code in this repository is structured in several folders:

- **actuators** – code to control actuators (i.e., smart speakers, lights, and humidifiers) to generate context stimuli (e.g., blinking light). 
- **data-collection** – code to collect sensor data (audio, illuminance, RGB, CO2, TVOC) using sensors attached to a Raspberry Pi. 
- **evaluation** – code to compute similarity and entropy of sensor data collected in our experiments.
- **visualization** – code to produce plots used in our paper based on the results generated with the evaluation code. 

# License
All code is licensed under the Apache 2.0, unless noted otherwise. See LICENSE for details.
