import os
import json
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from threading import Thread
from scipy.io import wavfile
import eval_metrics


time_format = "%Y-%m-%d %H:%M:%S.%m"


def list_files(file_path, file_format):
    return glob(os.path.join(os.path.join(file_path, "**"), file_format), recursive=True)


def multi_entro(results, experiment, save_path):
    bins = [5, 10, 20, 25, 50, 500, 750, 1000, 2000, 5000, 10000, 25000, 50000]
    for b in bins:
        print(f"{experiment} at bin: {b}")
        eval_metrics.entropy_matrix(results[experiment], f"Entropy {experiment} {b}", save_path=save_path, bin_size=b,
                                    plot_hist=True, is_audio=True)


def plot_waveform(data, sampling_rate, file_path, experiment, device):
    n_samples = np.arange(0, len(data) / sampling_rate, 1 / sampling_rate).astype("timedelta64[m]")
    save_path = os.path.join(file_path, "audio", experiment)
    os.makedirs(save_path, exist_ok=True)

    caption = f"{experiment} for {device}"
    plt.plot(n_samples, data)
    plt.title(f"{experiment} device: {device}")
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    print(f"draw image for {experiment} {device}")
    plt.savefig(os.path.join(save_path, caption + ".png"))
    plt.clf()
    plt.close("all")
    print("image done")


def calc_entropy(file_path, save_path=False, multithreading=False, plot_hist=False, plot_wform=False):
    # Get data files from directory
    print("Computing entropy of the data in '*.wav' files inside: '%s'" % file_path)
    print()
    files = list_files(file_path, "*.wav")
    files.sort()

    # Store results for audio entropy calculation
    results = {}
    entropy = {"audio": {}}

    # We assume data  is stored as follows: .../Experiment/Sensor-Nr/xx.wav, e.g.,
    # .../Home/Sensor-01/01.wav
    for file in files:
        if os.sys.platform == "win32":
            experiment = os.path.basename(os.path.dirname(os.path.dirname(file)))
            device = os.path.basename(os.path.dirname(file))
        else:
            experiment = os.path.basename(os.path.dirname(os.path.dirname(file)))
            device = os.path.basename(os.path.dirname(file))

        # Load audio file
        sampling_rate, data = wavfile.read(file)

        # Check if we want to print audio waveform for analysis
        if plot_wform:
            plot_waveform(data, sampling_rate, file_path, experiment, device)
        else:
            if not results.__contains__(experiment):
                results[experiment] = []
            results[experiment].append(list((data, device)))

    for experiment in results:
        if not entropy["audio"].__contains__(experiment):
            entropy["audio"][experiment] = {}
        print(f"Entropy for experiment {experiment}")

        # Check if we want to generate plots or not
        if save_path:
            save_path = os.path.join(file_path, "audio", experiment)
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = None

        if plot_hist:
            input(f"Show {experiment} ?")
            for dev_data, dev_name in results[experiment]:
                plt.title(f"{experiment} - {dev_name}")
                plt.hist(dev_data, "fd")
                plt.show()
                plt.close()
        else:
            if multithreading:
                threads = [Thread(target=multi_entro, kwargs=dict(resultDict=results, save_path=save_path,
                                                                  experiment=experiment))]
                for t in threads:
                    t.start()
                    print("thread started")
                for t in threads:
                    t.join()
                print(f"{experiment} all threads finished")
                return
            else:
                # Number of bins considered for computing entropy of audio data (we need 15)
                bins = [10, 15, 20, 30, 50, 500, 1000, 1500, 5000, 10000, 15000, 30000, 60000]
                for b in bins:
                    print(f"{experiment} at bin: {b}")
                    entropy["audio"][experiment][b] = eval_metrics.entropy_matrix(results[experiment],
                                                                                  f"Entropy {experiment} {b}",
                                                                                  save_path=save_path,
                                                                                  bin_size=b,
                                                                                  plot_hist=False,
                                                                                  is_audio=True)

    # Save calculated entropy in JSON
    with open(os.path.join(file_path, "audio_entropy.json"), "w") as json_file:
        json.dump(entropy, json_file)
