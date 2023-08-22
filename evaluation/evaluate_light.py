import os
from glob import glob
import date_time_format
import numpy as np
import eval_metrics
import json


def file_to_dict(files, min_split=None):
    # Store file data in a dictionary
    files_data = {}

    # Iterate over files
    for file in files:
        # Store light values (1, 2, 3) and timestamps
        timestamp = []
        ld1 = []
        ld2 = []
        ld3 = []

        device = os.path.basename(os.path.dirname(file))
        dir_path = os.path.dirname(os.path.dirname(file))
        distance = os.path.basename(os.path.dirname(os.path.dirname(file)))

        if not files_data.keys().__contains__(distance):
            files_data[distance] = {}
            if min_split is None:
                files_data[distance]["path"] = os.path.join(dir_path, "results")
            else:
                files_data[distance]["path"] = os.path.join(dir_path, f"results{min_split}light")

        # Open a file
        with open(file, "r") as f:
            while True:
                tmp = f.readline()
                if tmp == "":
                    break

                # Get timestamp and data
                tmp = tmp.replace(",", "")
                ts, d1, d2, d3 = tmp.split(" ")
                timestamp.append(date_time_format.get_date(ts))

                ld1.append(int(float(d1)))
                ld2.append(int(float(d2)))
                ld3.append(int(float(d3)))

        files_data[distance][device] = (timestamp, ld1, ld2, ld3)

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
        split_size.append(split)
        step = round(len(ts) / split)
        interval = np.arange(0, len(ts), step)

        if len(interval) <= split:
            interval = np.append(interval, len(ts) - 1)  # add the last chunk to the array
        time_split_index[device] = interval

    return time_split_index, int(np.min(split_size)) - 1


def evaluate(file_path, minutes_split=10, filter_data=True, normalize_data=False):
    # Get data files from directory
    print("Computing DTW distance of the data in 'RGB.txt' files inside: '%s'" % file_path)
    print()
    files = glob(os.path.join(os.path.join(file_path, "**"), "RGB.txt"), recursive=True)
    files.sort()

    # Store dtw distance
    dtw_dist = {}

    # Convert data files for each experiment to a dictionary
    results = file_to_dict(files, minutes_split)

    # Iterate over experiments
    for experiment in results:
        # Store RGB and illuminance data + timestamps
        timestamp = []
        rgb = []
        rgb_raw = []
        lum = []
        lum_raw = []

        for device in results[experiment]:
            if device == "path":
                continue

            timestamp.append(list((results[experiment][device][0], device)))

            # Sensor data normalization and filtering
            if not normalize_data:
                rgb.append(list((eval_metrics.filter_1(eval_metrics.normalize_mean_sub(results[experiment][device][2])),
                                 device)))
                rgb_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][2]), device)))
                lum.append(list((eval_metrics.filter_1(eval_metrics.normalize_mean_sub(results[experiment][device][1])),
                                 device)))
                lum_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][1]), device)))
            else:
                rgb.append(list((eval_metrics.filter_1(results[experiment][device][2]), device)))
                rgb_raw.append(list((results[experiment][device][2], device)))
                lum.append(list((eval_metrics.filter_1(results[experiment][device][1]), device)))
                lum_raw.append(list((results[experiment][device][1], device)))

        start_time = timeline_index_start(timestamp)
        end_time = timeline_index_end(timestamp)
        index = 0

        # Trim all data to the same time length
        for d in start_time:
            trim_to_index(timestamp[index], start_time[d], end_time[d])
            trim_to_index(rgb[index], start_time[d], end_time[d])
            trim_to_index(lum[index], start_time[d], end_time[d])
            index += 1

        dtw_dist[experiment] = {}

        # In case of chunking
        if minutes_split is not None:
            chunk_index, split_size = split_into_time_chunks(timestamp, minutes=minutes_split)

            # Iterate over chunks
            for i in range(split_size):
                rgb_chunk = []
                lum_chunk = []
                split_index = 0

                for d in chunk_index:  # trim to sub_lists
                    start_index = chunk_index[d][i]
                    end_index = chunk_index[d][i + 1]
                    to_add_rgb = rgb[split_index][0][start_index:end_index]
                    to_add_lum = lum[split_index][0][start_index:end_index]

                    # Normalization for chunks
                    if normalize_data:
                        to_add_rgb = eval_metrics.normalize_mean_sub(to_add_rgb)
                        to_add_lum = eval_metrics.normalize_mean_sub(to_add_lum)

                    # Filtering for chunks
                    if filter_data:
                        to_add_rgb = eval_metrics.filter_2(to_add_rgb)
                        to_add_lum = eval_metrics.filter_2(to_add_lum)

                    rgb_chunk.append(list((to_add_rgb, d)))
                    lum_chunk.append(list((to_add_lum, d)))
                    split_index += 1

                # Per chunk DTW distance
                dtw_dist[experiment][(i + 1) * minutes_split] = {}

                # Compute DTW distance for RGB
                tab, sim_mat, _ = eval_metrics.similarity_matrix(rgb_chunk,
                                                                 caption=f"Distance Matrix RGB {(i + 1) * minutes_split}min")
                dtw_dist[experiment][(i + 1) * minutes_split]["RGB"] = sim_mat

                # Compute DTW distance for illuminance
                tab, sim_mat, _ = eval_metrics.similarity_matrix(lum_chunk,
                                                                 caption=f"Distance Matrix LUM {(i + 1) * minutes_split}min")
                dtw_dist[experiment][(i + 1) * minutes_split]["lum"] = sim_mat

        else:
            for i in range(len(rgb)):
                # Filtering for full data
                if filter_data:
                    rgb[i][0] = eval_metrics.filter_2(rgb[i][0])
                    lum[i][0] = eval_metrics.filter_2(lum[i][0])
                else:
                    rgb[i][0] = rgb[i][0]
                    lum[i][0] = lum[i][0]

                # Normalization for full data
                if normalize_data:
                    rgb[i][0] = eval_metrics.filter_2(eval_metrics.normalize_mean_sub(rgb[i][0]))
                    lum[i][0] = eval_metrics.filter_2(eval_metrics.normalize_mean_sub(lum[i][0]))

            # DTW distance for full sensor recording
            dtw_dist[experiment]["full"] = {}

            # Compute DTW distance for RGB
            tab, sim_mat, _ = eval_metrics.similarity_matrix(rgb, caption="Distance Matrix RGB")
            dtw_dist[experiment]["full"]["RGB"] = sim_mat

            # Compute DTW distance for illuminance
            tab, sim_mat, _ = eval_metrics.similarity_matrix(lum, caption="Distance Matrix LUM")
            dtw_dist[experiment]["full"]["lum"] = sim_mat

        print(f"{experiment} Done in Light")

    # Pick resulting file name: either chunk or full data recording
    if minutes_split is None:
        name = "full"
    else:
        name = minutes_split

    # Save computed DTW distance in JSON
    with open(os.path.join(file_path, f"distance_metrics_light_{name}.json"), "w") as f:
        json.dump(dtw_dist, f, indent=4, sort_keys=True)


def get_min_max_values(results, filter_data, skip_device=" "):
    r_max = l_max = 0
    r_min = l_min = float("inf")

    # For each experiment iterate over sensor data of each device
    for experiment in results:
        for device in results[experiment]:
            if device == "path":
                continue

            if device.__contains__(skip_device):
                continue

            # Check if we need to do filtering
            if filter_data:
                l_max = max(l_max, np.max(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][1])))))
                l_min = min(l_min, np.min(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][1])))))
                r_max = max(r_max, np.max(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][2])))))
                r_min = min(r_min, np.min(
                    (eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][2])))))
            else:
                l_max = max(l_max, np.max(eval_metrics.normalize_mean_sub(results[experiment][device][1])))
                l_min = min(l_min, np.min(eval_metrics.normalize_mean_sub(results[experiment][device][1])))
                r_max = max(r_max, np.max(eval_metrics.normalize_mean_sub(results[experiment][device][2])))
                r_min = min(r_min, np.min(eval_metrics.normalize_mean_sub(results[experiment][device][2])))

    return r_max, r_min, l_max, l_min


def calc_entropy(file_path, save_path=False):
    # Get data files from directory
    print("Computing entropy of the data in 'RGB.txt' files inside: '%s'" % file_path)
    print()
    files = glob(os.path.join(os.path.join(file_path, "**"), "RGB.txt"), recursive=True)
    files.sort()

    # Convert data files for each experiment to a dictionary
    results = file_to_dict(files)

    modalities = ["RGB", "RGB raw", "lum", "lum raw"]
    entropy = {}

    # Populate a dict with keys
    for mod in modalities:
        entropy[mod] = {}

    r_max, r_min, l_max, l_min = get_min_max_values(results, filter_data=False)
    r_max_f, r_min_f, l_max_f, l_min_f = get_min_max_values(results, filter_data=True)

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

        # Store RGB and illuminance data
        timestamp = []
        rgb = []
        rgb_raw = []
        lum = []
        lum_raw = []

        # Iterate over sensor data of each device
        for device in results[experiment]:
            if device == "path":
                continue

            timestamp.append(list((results[experiment][device][0], device)))
            rgb.append(list((eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][2])),
                             device)))
            rgb_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][2]), device)))
            lum.append(list((eval_metrics.normalize_mean_sub(eval_metrics.filter_1(results[experiment][device][1])),
                             device)))
            lum_raw.append(list((eval_metrics.normalize_mean_sub(results[experiment][device][1]), device)))

        # Number of bins considered for computing entropy of RGB and illuminance data (we need 20 and 100)
        bins = [5, 20, 25, 30, 35, 40, 45, 50, 75, 100, 200]

        for b in bins:
            for mod in modalities:
                entropy[mod][experiment][b] = {}

            caption_rgb = f"entropy RGB {b}"
            caption_rgb_raw = f"entropy RGB raw {b}"
            caption_lum = f"entropy lum {b}"
            caption_lum_raw = f"entropy lum raw {b}"

            if experiment.__contains__("Home"):
                rgb_mod = eval_metrics.get_modality_type("rgb_home")
            else:
                rgb_mod = eval_metrics.get_modality_type("rgb_office")

            lux_mod = eval_metrics.get_modality_type("lum")

            entropy[modalities[0]][experiment][b] = eval_metrics.entropy_matrix(rgb, caption_rgb,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(r_max_f, r_min_f),
                                                                                modality_type=rgb_mod)

            entropy[modalities[1]][experiment][b] = eval_metrics.entropy_matrix(rgb_raw, caption_rgb_raw,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(r_max, r_min),
                                                                                modality_type=rgb_mod)

            entropy[modalities[2]][experiment][b] = eval_metrics.entropy_matrix(lum, caption_lum,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(l_max_f, l_min_f),
                                                                                modality_type=lux_mod)

            entropy[modalities[3]][experiment][b] = eval_metrics.entropy_matrix(lum_raw, caption_lum_raw,
                                                                                save_path=save_path,
                                                                                bin_size=b,
                                                                                border_bins=(l_max, l_min),
                                                                                modality_type=lux_mod)

    # Save calculated entropy in JSON
    with open(os.path.join(file_path, "light_entropy.json"), "w") as json_file:
        json.dump(entropy, json_file)
