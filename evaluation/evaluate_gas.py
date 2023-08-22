import os
from glob import glob
import date_time_format
import numpy as np
import eval_metrics
import json

eof = b"\x00".decode()


def list_files(file_path, file_format):
    return glob(os.path.join(os.path.join(file_path, "**"), file_format), recursive=True)


def read_gas_file(file, files_data, min_split=None):
    # Store CO2 and TVOC data + timestamps
    tvoc_data = []
    co2_data = []
    timestamp = []

    # Open a file for reading
    with open(file, "r") as f:
        device = os.path.basename(os.path.dirname(file))
        dir_path = os.path.dirname(os.path.dirname(file))
        distance = os.path.basename(dir_path)

        if not files_data.__contains__(distance):
            files_data[distance] = {}
            if min_split is None:
                files_data[distance]["path"] = os.path.join(dir_path, "results")
            else:
                files_data[distance]["path"] = os.path.join(dir_path, f"results{min_split}gas")

        store_data = files_data[distance]

        while True:
            tmp = f.readline()
            if tmp == "" or tmp == eof:
                break

            # Do some formatting
            tmp = tmp.replace("ppm", "")
            tmp = tmp.replace("eCO2 =", "")
            tmp = tmp.replace("TVOC =", "")
            tmp = tmp.replace("ppb", "")

            ts, co2, tvoc = tmp.split("\t")
            co2 = int(co2)
            # Last element has eof b"\x00" appended which is somehow not removed by strip etc
            tvoc = int(tvoc.replace(eof, ""))

            timestamp.append(date_time_format.get_date(ts))
            co2_data.append(co2)
            tvoc_data.append(tvoc)

        co2_data = np.array(co2_data)
        tvoc_data = np.array(tvoc_data)
        timestamp = np.array(timestamp)

        store_data[device] = (timestamp, co2_data, tvoc_data)

        return files_data


def timeline_index_start(timestamps):
    min_ts = []

    for ts, device in timestamps:
        min_ts.append(ts[0])

    max_ts = max(min_ts)
    start_index = {}

    for ts, device in timestamps:
        index = 0
        for date in ts:
            if date >= max_ts:
                start_index[device] = index
                break
            index += 1

    return start_index


def timeline_index_end(timestamps):
    max_ts = []

    for ts, device in timestamps:
        max_ts.append(max(ts))

    min_ts = min(max_ts)
    start_index = {}

    for ts, device in timestamps:
        index = 0
        for date in reversed(ts):
            if date <= min_ts:
                start_index[device] = len(ts) - index
                break
            index += 1

    return start_index


def trim_to_index(data, start_index, end_index):
    data[0] = data[0][start_index:end_index]


def split_into_time_chunks(timestamps, minutes=10):
    split_size = []
    time_split_index = {}

    for ts, device in timestamps:
        split = (ts[-1] - ts[0]).seconds / (minutes * 60)
        step = round(len(ts) / split)
        interval = np.arange(0, len(ts), step)

        if round(split - round(split)):
            interval = np.append(interval, len(ts) - 1)  # add the last chunk to the array

        split_size.append(len(interval))
        time_split_index[device] = interval

    return time_split_index, int(np.min(split_size)) - 1


def evaluate(file_path, minutes_split=10, filter_data=True, normalize_data=False):
    # Get data files from directory
    print("Computing DTW distance of the data in 'gas.txt' files inside: '%s'" % file_path)
    print()
    files = glob(os.path.join(os.path.join(file_path, "**"), "gas.txt"), recursive=True)
    files.sort()

    # Store dtw distance
    dtw_dist = {}

    # Store data for each experiment in the scenario, i.e., base, attacker, entropy, entropy_attacker
    results = {}
    for file in files:
        read_gas_file(file, results, minutes_split)

    # Iterate over experiments
    for experiment in results:
        # Store CO2 and TVOC data + timestamps
        timestamp = []
        co2 = []
        co2_raw = []
        tvoc = []
        tvoc_raw = []

        for device in results[experiment]:
            if device == "path":
                continue

            timestamp.append(list((results[experiment][device][0], device)))

            # Sensor data normalization and filtering
            if not normalize_data:
                co2.append(list((eval_metrics.filter_1(eval_metrics.normalize_mean_sub(results[experiment][device][1])),
                                 device)))
                co2_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][1]), device)))
                tvoc.append(list((eval_metrics.filter_1(eval_metrics.normalize_mean_sub(results[experiment][device][2])),
                                  device)))
                tvoc_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][2]), device)))
            else:
                co2.append(list((eval_metrics.filter_1(results[experiment][device][1]), device)))
                co2_raw.append(list((results[experiment][device][1], device)))
                tvoc.append(list((eval_metrics.filter_1(results[experiment][device][2]), device)))
                tvoc_raw.append(list((results[experiment][device][2], device)))

        start_time = timeline_index_start(timestamp)
        end_time = timeline_index_end(timestamp)
        index = 0

        # Trim all data to the same time length
        for d in start_time:
            trim_to_index(timestamp[index], start_time[d], end_time[d])
            trim_to_index(co2[index], start_time[d], end_time[d])
            trim_to_index(tvoc[index], start_time[d], end_time[d])
            index += 1

        dtw_dist[experiment] = {}

        # In case of chunking
        if minutes_split is not None:
            chunk_index, split_size = split_into_time_chunks(timestamp, minutes=minutes_split)

            # Iterate over chunks
            for i in range(split_size - 1):
                co2_chunk = []
                tvoc_chunk = []
                split_index = 0

                for d in chunk_index:  # trim to sub_lists
                    start_index = chunk_index[d][i]
                    end_index = chunk_index[d][i + 1]
                    to_add_co2 = co2[split_index][0][start_index:end_index]
                    to_add_tvoc = tvoc[split_index][0][start_index:end_index]

                    # Normalization for chunks
                    if normalize_data:
                        to_add_co2 = eval_metrics.normalize_mean_sub(to_add_co2)
                        to_add_tvoc = eval_metrics.normalize_mean_sub(to_add_tvoc)

                    # Filtering for chunks
                    if filter_data:
                        to_add_co2 = eval_metrics.filter_2(to_add_co2)
                        to_add_tvoc = eval_metrics.filter_2(to_add_tvoc)

                    co2_chunk.append(list((to_add_co2, d)))
                    tvoc_chunk.append(list((to_add_tvoc, d)))
                    split_index += 1

                # Per chunk DTW distance
                dtw_dist[experiment][(i + 1) * minutes_split] = {}

                # Compute DTW distance for CO2
                tab, sim_mat, _ = eval_metrics.similarity_matrix(co2_chunk,
                                                                 caption=f"Distance Matrix CO2{(i + 1) * minutes_split}min")
                dtw_dist[experiment][(i + 1) * minutes_split]["CO2"] = sim_mat

                # Compute DTW distance for TVOC
                tab, sim_mat, _ = eval_metrics.similarity_matrix(tvoc_chunk,
                                                                 caption=f"Distance Matrix TVOC{(i + 1) * minutes_split}min")
                dtw_dist[experiment][(i + 1) * minutes_split]["TVOC"] = sim_mat

        else:
            for i in range(len(co2)):
                # Filtering for full data
                if filter_data:
                    co2[i][0] = eval_metrics.filter_2(co2[i][0])
                    tvoc[i][0] = eval_metrics.filter_2(tvoc[i][0])
                else:
                    co2[i][0] = co2[i][0]
                    tvoc[i][0] = tvoc[i][0]

                # Normalization for full data
                if normalize_data:
                    co2[i][0] = eval_metrics.filter_2(eval_metrics.normalize_mean_sub(co2[i][0]))
                    tvoc[i][0] = eval_metrics.filter_2(eval_metrics.normalize_mean_sub(tvoc[i][0]))

            # DTW distance for full sensor recording
            dtw_dist[experiment]["full"] = {}

            # Compute DTW distance for CO2
            tab, sim_mat, _ = eval_metrics.similarity_matrix(co2, caption="Distance Matrix CO2")
            dtw_dist[experiment]["full"]["CO2"] = sim_mat

            # Compute DTW distance for TVOC
            tab, sim_mat, _ = eval_metrics.similarity_matrix(tvoc, caption="Distance Matrix TVOC")
            dtw_dist[experiment]["full"]["TVOC"] = sim_mat

        print(f"{experiment} Done in Gas")

    # Pick resulting file name: either chunk or full data recording
    if minutes_split is None:
        name = "full"
    else:
        name = minutes_split

    # Save computed DTW distance in JSON
    with open(os.path.join(file_path, f"distance_metrics_gas_{name}.json"), "w") as f:
        json.dump(dtw_dist, f, indent=4, sort_keys=True)


def get_min_max_values(results, filter_data, skip_device=" "):
    c_max = t_max = 0
    c_min = t_min = float("inf")

    # For each experiment iterate over sensor data of each device
    for experiment in results:
        for device in results[experiment]:

            if device == "path":
                continue

            if device.__contains__(skip_device):
                continue

            # Check if we need to do filtering
            if filter_data:
                c_max = max(c_max, np.max(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][1])))))
                c_min = min(c_min, np.min(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][1])))))
                t_max = max(t_max, np.max(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][2])))))
                t_min = min(t_min, np.min(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][2])))))
            else:
                c_max = max(c_max, np.max(results[experiment][device][1]))
                c_min = min(c_min, np.min(results[experiment][device][1]))
                t_max = max(t_max, np.max(results[experiment][device][2]))
                t_min = min(t_min, np.min(results[experiment][device][2]))

    return c_max, c_min, t_max, t_min


def calc_entropy(file_path, save_path=False):
    # Get data files from directory
    print("Computing entropy of the data in 'gas.txt' files inside: '%s'" % file_path)
    print()
    files = list_files(file_path, "gas.txt")
    files.sort()

    # Store data for each experiment in the scenario, i.e., base, attacker, entropy, entropy_attacker
    results = {}
    for file in files:
        read_gas_file(file, results)

    modalities = ["CO2", "CO2 raw", "TVOC", "TVOC raw"]
    entropy = {}

    # Populate entropy dictionary with keys
    for mod in modalities:
        entropy[mod] = {}

    c_max, c_min, t_max, t_min = get_min_max_values(results, filter_data=False)
    c_max_f, c_min_f, t_max_f, t_min_f = get_min_max_values(results, filter_data=True)

    # Iterate over experiments
    for experiment in results:
        # Iterate over modalities to set dictionary keys
        for mod in modalities:
            entropy[mod][experiment] = {}

        # Check if we want to generate plots or not
        if save_path:
            save_path = os.path.join(file_path, "resultsEntropy", experiment)
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = None

        # Store CO2 and TVOC data + timestamps
        timestamp = []
        co2 = []
        co2_raw = []
        tvoc = []
        tvoc_raw = []

        # Iterate over sensor data of each device
        for device in results[experiment]:
            if device == "path":
                continue

            timestamp.append(list((results[experiment][device][0], device)))
            co2.append(list((eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][1])),
                             device)))
            co2_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][1]), device)))
            tvoc.append(list((eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][2])),
                              device)))
            tvoc_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][2]), device)))

        # Number of bins considered for computing entropy of CO2 and TVOC data (we need 5 and 20)
        bins = [5, 20, 23, 25, 30, 35, 40, 45, 50, 75, 100]

        for b in bins:
            for mod in modalities:
                entropy[mod][experiment][b] = {}

            caption_co2 = f"entropy CO2 {b}"
            caption_co2_raw = f"entropy CO2 raw {b}"
            caption_tvoc = f"entropy TVOC {b}"
            caption_tvoc_raw = f"entropy TVOC raw {b}"

            co2_mod = eval_metrics.get_modality_type("co2")
            tvoc_mod = eval_metrics.get_modality_type("tvoc")

            entropy[modalities[0]][experiment][b] = eval_metrics.entropy_matrix(co2, caption_co2,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(c_max_f, c_min_f),
                                                                                modality_type=co2_mod)

            entropy[modalities[1]][experiment][b] = eval_metrics.entropy_matrix(co2_raw, caption_co2_raw,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(c_max, c_min),
                                                                                modality_type=co2_mod)

            entropy[modalities[2]][experiment][b] = eval_metrics.entropy_matrix(tvoc, caption_tvoc,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(t_max_f, t_min_f),
                                                                                modality_type=tvoc_mod)

            entropy[modalities[3]][experiment][b] = eval_metrics.entropy_matrix(tvoc_raw, caption_tvoc_raw,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(t_max, t_min),
                                                                                modality_type=tvoc_mod)

    # Save calculated entropy in JSON
    with open(os.path.join(file_path, "gas_entropy.json"), "w") as json_file:
        json.dump(entropy, json_file)
