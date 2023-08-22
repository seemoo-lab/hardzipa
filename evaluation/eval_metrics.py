from scipy.stats import entropy
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import dtw
from terminaltables import AsciiTable


def filter_1(data):
    return gaussian_filter(savgol_filter(data, 3, 2), sigma=1.4)


def filter_2(data):
    return gaussian_filter(savgol_filter(data, 5, 3), sigma=1.4)


def normalize_mean_sub(data):
    return np.array(data) - np.mean(data)


def dynamic_time_wrapping(data1, data2, xlab="xlab", ylab="ylab", draw=False):
    # Compute DTW distance between two time series: data1 and data2
    alignment = dtw.dtw(data1, data2, keep_internals=False, distance_only=True)
    dtw_dist = alignment.distance

    # Check if we want to plot DTW
    if draw:
        alignment.plot(type="twoway", xlab=xlab, ylab=ylab)

    return dtw_dist


def scpy_entropy(data):
    return entropy(data, base=2)


def get_modality_type(mod):
    if mod == "co2":
        return 1
    if mod == "tvoc":
        return 2
    if mod == "lum":
        return 3
    if mod == "rgb_home":
        return 4
    if mod == "rgb_office":
        return 5
    return 0


def get_max_bins(data, bins=None, is_audio=False, audio_pcm=16, max_range=None, min_range=None, modality_type=0,
                 min_max_scaling=False):
    # Case for audio data
    if is_audio:
        min_edge = (2 ** (audio_pcm - 1)) * -1
        max_edge = (2 ** (audio_pcm - 1)) - 1
        a_min = -2000
        a_max = 2000
        bottom_outlier = np.histogram_bin_edges((), bins=int(4), range=(min_edge, a_min))[:-1]
        top_outlier = np.histogram_bin_edges((), bins=int(4), range=(a_max, max_edge))[1:]
        bins = np.insert((np.histogram_bin_edges((), bins=bins, range=(a_min, a_max))), 0, bottom_outlier)
        print(f"bottom \n {bottom_outlier} \n top \n {top_outlier}")
        bins = np.rint(np.append(bins, top_outlier))
        return bins

    if min_range is None:
        min_range = np.min(data[0][0])
    if max_range is None:
        max_range = np.max(data[0][0])

    min_range = min(np.min(data), min_range)
    max_range = max(np.max(data), max_range) + 1

    base = np.mean(data)

    if get_modality_type("co2") == modality_type:  # (-5,20)
        if min_max_scaling:
            b = np.histogram_bin_edges((), bins=bins, range=(0, 100 + 1))
        b = np.sort(np.append(np.insert(np.histogram_bin_edges((), bins=bins, range=(base - 5, base + 20 + 1)), 0,
                                        min_range), max_range))

    if get_modality_type("tvoc") == modality_type:  # (-10, 20)
        if min_max_scaling:
            b = np.histogram_bin_edges((), bins=bins, range=(0, 100))
        b = np.sort(np.append(np.insert(np.histogram_bin_edges((), bins=bins, range=(base - 10, base + 20 + 1)), 0,
                                        min_range), max_range))

    if get_modality_type("lum") == modality_type:  # (-20, 20)
        if min_max_scaling:
            b = np.histogram_bin_edges((), bins=bins, range=(0, 100))
        b = np.sort(np.append(np.insert(np.histogram_bin_edges((), bins=bins, range=(base - 20, base + 20 + 1)), 0,
                                        min_range), max_range))

    if get_modality_type("rgb_home") == modality_type:  # (-90, 100)
        if min_max_scaling:
            b = np.histogram_bin_edges((), bins=bins, range=(0, 100))
        b = np.sort(np.append(np.insert(np.histogram_bin_edges((), bins=bins, range=(base - 2500, base + 3000 + 1)), 0,
                                        min_range), max_range))

    if get_modality_type("rgb_office") == modality_type:  # (-90, 100)
        if min_max_scaling:
            b = np.histogram_bin_edges((), bins=bins, range=(0, 100 + 1))
        b = np.sort(np.append(np.insert(np.histogram_bin_edges((), bins=bins, range=(base - 90, base + 100 + 1)), 0,
                                        min_range), max_range))

    if modality_type != 0:
        if np.min(data) < np.min(b):
            b = np.insert(b, 0, np.min(data))
        if np.max(data) > np.max(b):
            b = np.append(b, np.max(data))
        return b

    if bins is not None:
        return np.histogram_bin_edges((), bins=bins, range=(min_range, max_range))

    data_bins = []
    bins_size = []

    for dev_data, dev_name in data:
        bins = np.histogram_bin_edges(dev_data, "auto", range=(min_range, max_range))
        data_bins.append(bins)
        bins_size.append(len(bins))

    selected_bins = bins_size.index(np.min(bins_size))

    return data_bins[selected_bins]


def probability_hist(data, bins):
    dist = np.histogram(data, bins=bins)
    return dist[0] / np.sum(dist[0])


def entropy_series(data, bins=20, is_audio=False, audio_pcm=16, border_bins=None, modality_type=0,
                   min_max_scaling=False):
    fixed_bin_size = bins

    # Check if we deal with audio data or other data
    if is_audio:
        audio_bins = get_max_bins(None, bins=bins, is_audio=is_audio, audio_pcm=audio_pcm, modality_type=modality_type)

    # Store entropy values and device names
    ent = []
    devices = []

    # Iterate over sensor data of each device
    for dev_data, dev_name in data:
        devices.append(dev_name)

        if not is_audio:
            if border_bins is None:
                bins = get_max_bins(dev_data, bins=bins, is_audio=is_audio, audio_pcm=audio_pcm,
                                    modality_type=modality_type, min_max_scaling=min_max_scaling)
            else:
                bins = get_max_bins(dev_data, bins=bins, is_audio=is_audio, audio_pcm=audio_pcm,
                                    max_range=border_bins[0], min_range=border_bins[1], modality_type=modality_type,
                                    min_max_scaling=min_max_scaling)
        else:
            bins = audio_bins

        bin_size = len(bins) - 1

        # Formula to compute entropy
        ent.append(scpy_entropy(probability_hist(dev_data, bins)) / np.log2(bin_size))
        bins = fixed_bin_size

    return ent, devices


def entropy_matrix(data, caption, save_path=None, bin_size=20, plot_hist=False, is_audio=False, audio_pcm=16,
                   border_bins=None, modality_type=0, min_max_scaling=False):
    # Compute entropy for the data collected by each device
    ent, devices = entropy_series(data, bins=bin_size, is_audio=is_audio, audio_pcm=audio_pcm, border_bins=border_bins,
                                  modality_type=modality_type, min_max_scaling=min_max_scaling)

    if plot_hist:
        for dev_data, dev_name in data:
            bins = get_max_bins(data, bins=bin_size, is_audio=is_audio, audio_pcm=audio_pcm,
                                modality_type=modality_type)
            plt.hist(dev_data, bins="auto")
            plt.title(f"Hist of {dev_name} with auto bins")
            plt.savefig(os.path.join(save_path, f"{dev_name}-{len(bins) - 1}"))

    res_ent = list(zip(ent, devices))

    # Plot the amount of entropy per device if necessary
    if save_path:
        # Store labels (X-axis values)
        labels = []

        for dev_data, dev_name in data:
            labels.append(dev_name)

        fig, ax = plt.subplots()
        ax.bar(np.arange(len(labels)), ent, width=0.15, label='Entropy', ecolor='black')

        ax.set_ylabel("Entropy")
        ax.set_xlabel("Device")
        ax.set_title(caption)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.savefig(os.path.join(save_path, caption + ".png"))
        plt.close("all")

        table = []
        labels.insert(0, "Entropy")
        table.append(labels)
        ent.insert(0, "entropy")
        table.append(ent)
        tab = AsciiTable(table, title=caption)

    return res_ent


def similarity_matrix(data, print_matrix=False, caption="Distance Matrix"):
    # Store DTW results here
    sim_matrix = {}
    table = []
    header = [" "]
    table.append(header)
    max_distance = 0

    # Iterate over data of device 1 and device 2 (pairwise comparison)
    for dev1_data, dev1_name in data:
        sim_matrix[dev1_name] = {}
        header.append(dev1_name)
        row = [dev1_name]
        table.append(row)

        for dev2_data, dev2_name in data:
            if dev1_name == dev2_name:
                row.append(str(0))
            elif len(header) <= len(row):
                dtw_distance = dynamic_time_wrapping(dev1_data, dev2_data, dev1_name, dev2_name)
                row.append(str(dtw_distance))
                sim_matrix[dev1_name][dev2_name] = dtw_distance
                max_distance = max(max_distance, dtw_distance)
            else:
                row.append(" - ")

    # Normalized similarity matrix
    sim_matrix_norm = {}

    for dev1_name in sim_matrix:
        sim_matrix_norm[dev1_name] = {}
        for dev2_name in sim_matrix[dev1_name]:
            sim_matrix_norm[dev1_name][dev2_name] = sim_matrix[dev1_name][dev2_name] / max_distance

    tab = AsciiTable(table, title=caption)
    if print_matrix:
        print(tab.table)

    return tab, sim_matrix, sim_matrix_norm
