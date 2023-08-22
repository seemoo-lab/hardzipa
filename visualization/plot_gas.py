import os
from glob import glob
import date_time_format
import numpy as np
import plot_metrics
import json
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.dates import DateFormatter
import datetime
import re
import sys

eof = b'\x00'.decode()


def generate_distance_plot_from_existing_files(file_path):
    # Read result files
    files = glob(os.path.join(file_path, '**', 'distance_metrics_gas_*'), recursive=True)

    # Iterate over result files
    for file in files:
        # Get file name
        file_name = os.path.basename(file)

        # Open file for reading
        with open(file, 'r') as json_file:
            # Get experiment name
            experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))

            # Get result data
            data = json.load(json_file)

            # Check if we need to set up a flag for the follow-up experiment
            if experiment.lower().__contains__('new'):
                is_not_followup = False
            else:
                is_not_followup = True

            # Get sensor modality
            modality = file_name.split('.')[0].split('_')[-2]

            # Get chunk interval
            match = re.search(modality + r'_(.*)\.json', file_name)

            # If there is no match exit
            if not match:
                print('generate_distance_plot_from_existing_files: could not extract chunk size from file: "%s"'
                      % file_name)
                sys.exit(0)

            chunk = match.group(1)

            # Construct plot's name
            caption = f'{experiment}_distance_{chunk}'

            # Exclude the experiments we don't want to show
            exclude_list = ['Humid_close', 'Humid_1m']

            # Path to save plots
            save_path = os.path.join(file_path, 'Gas', 'distance', 'Plot_distance')

            # Create directory for saving plots
            os.makedirs(save_path, exist_ok=True)

            # Do some plotting
            plot_metrics.draw_experiment_distance_bar_plot(data, save_path=save_path, caption=caption, normalize=True,
                                                           exclude_list=exclude_list, arrange_order=is_not_followup,
                                                           is_gas_followup=not is_not_followup)


def generate_entropy_from_existing_files(file_path):
    # Read result files
    files = glob(os.path.join(file_path, '**', 'gas_entropy.json'), recursive=True)

    # Iterate over result files
    for file in files:
        # Open json file for reading
        with open(file, 'r') as json_file:
            # Get experiment name
            experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))

            # Get JSON data
            data = json.load(json_file)

            # Check if we need to set up a flag for the follow-up experiment
            if experiment.lower().__contains__('new'):
                is_not_followup = False
            else:
                is_not_followup = True

            # List for excluding some experiments, modality names, and bin sizes to evaluate entropy
            exclude_list = ['Humid_close', 'Humid_1m']
            modality = ['CO2', 'TVOC']
            bins = ['5', '20']

            # Path to save plots
            save_path = os.path.join(os.path.dirname(file), 'Plot_entropy')

            # Create directory for saving plots
            os.makedirs(save_path, exist_ok=True)

            # Iterate over modalities
            for mod in modality:
                # Iterate over bin sizes
                for b in bins:
                    # Do some plotting
                    plot_metrics.draw_experiment_entropy_bar_plot(data, save_path, f'{experiment}_entropy_{mod}',
                                                                  b, mod, exclude_list=exclude_list,
                                                                  arrange_order=is_not_followup,
                                                                  is_gas_followup=not is_not_followup)


def glob_files(path, pattern):
    return glob(os.path.join(os.path.join(path, '**'), pattern), recursive=True)


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
                files_data[distance]['path'] = os.path.join(dir_path, 'results')
            else:
                files_data[distance]['path'] = os.path.join(dir_path, f'results{min_split}gas')

        store_data = files_data[distance]

        while True:
            tmp = f.readline()
            if tmp == '' or tmp == eof:
                break

            # Do some formatting
            tmp = tmp.replace('ppm', '')
            tmp = tmp.replace('eCO2 =', '')
            tmp = tmp.replace('TVOC =', '')
            tmp = tmp.replace('ppb', '')

            ts, co2, tvoc = tmp.split("\t")
            co2 = int(co2)
            # Last element has eof b"\x00" appended which is somehow not removed by strip etc
            tvoc = int(tvoc.replace(eof, ''))

            timestamp.append(date_time_format.get_date(ts))
            co2_data.append(co2)
            tvoc_data.append(tvoc)

        co2_data = np.array(co2_data)
        tvoc_data = np.array(tvoc_data)
        timestamp = np.array(timestamp)

        store_data[device] = (timestamp, co2_data, tvoc_data)

        return files_data


def plot_gas_sample(data, time, caption, save_path):
    # Set this up to avoid errors in submission systems such as HotCRP (i.e., proper font embeddings)
    matplotlib.rcParams.update({'pdf.fonttype': 42})
    matplotlib.rcParams.update({'ps.fonttype': 42})

    # Set figure size
    plt.figure(figsize=(6.5, 4))

    # Adjust font for 10^x
    plt.rc('font', size=18)

    # Calculate the X labels and write timedelta back to datetime so that the DateFormatter works
    time_deltas = time - time[0]
    zero = datetime.datetime(2021, 1, 1)

    # Here we write it back so that we get nice minutes to display
    new_time = [zero + t for t in time_deltas]

    # Tell the formatter how to format only minutes
    if os.sys.platform.__contains__('win'):
        date_form = DateFormatter('%#M')
    else:
        date_form = DateFormatter('%-M')

    # Do the plotting, the color is called 'teal' from here: https://xkcd.com/color/rgb/
    plt.plot(new_time, data, label='CO2', color='#ac1db8', linewidth=6)

    # Add 10^x for ticks (Y-axis)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # Set X-axis
    plt.gca().xaxis.set_major_formatter(date_form)

    # Set font sizes of axes and their ticks
    plt.xlabel('Time (minutes)', fontsize=26)
    plt.ylabel('Concentration (ppm)', fontsize=26)
    plt.tick_params(axis='both', labelsize=22)

    # Add yticks
    plt.yticks(np.arange(4, 6.1, step=1) * 100)

    # Have ticks on left and right side of Y-axis
    plt.gca().yaxis.set_ticks_position('both')

    # Ticks inside the plot look nicer IMHO
    plt.tick_params(axis='y', direction='in')

    # Set up a legend
    lgnd = plt.legend(loc='best', ncol=1, borderaxespad=0.,
                      fontsize=21, handlelength=1, handletextpad=0.4)

    # Increase line thickness of legend markers
    for line in lgnd.get_lines():
        line.set_linewidth(7)

    # Grid
    plt.grid(True, axis='y', zorder=0)

    # Save or show the plot
    if save_path is not None:
        plt.savefig(os.path.join(save_path, caption + '.pdf'), format='pdf', dpi=1000, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, caption + '.svg'), format='svg', dpi=1000, bbox_inches='tight')
    else:
        plt.show()

    # Close figure
    plt.close('all')


def get_index_for_time(time_series, start, stop):
    start_index = 0

    for t in time_series:
        if t.strftime('%H:%M') == start:
            start_index = np.where(time_series == t)[0][0]
            break

    stop_index = len(time_series)

    for i in range(1, stop_index):
        if time_series[stop_index-i].strftime('%H:%M') == stop:
            stop_index -= i
            break

    return start_index, stop_index


def generate_example_change(file_path):
    # Read data files
    files = glob_files(file_path, 'gas.txt')
    results = {}

    # Iterate over data files
    for file in files:
        experiment = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file))))
        if not results.__contains__(experiment):
            results[experiment] = {}

        read_gas_file(file, results[experiment])

    # Pick data chunk to plot
    device2 = results['FreshCalib']['onOffDiff']['102']
    save_path = file_path
    start_index, stop_index = get_index_for_time(device2[0], '15:45', '16:00')

    plot_gas_sample(device2[1][start_index:stop_index], device2[0][start_index:stop_index], 'co2-humidifier',
                    save_path=save_path)


if __name__ == '__main__':
    # Provide path where results are stored
    # Get the results from: https://zenodo.org/record/8263497
    filepath = 'C:/Users/mfomichev/Desktop/hardzipa-results'

    # Get paths for each scenario --> case when we want to cover all scenarios at once
    # experiments = [os.path.join(filepath, 'Home'),
    #                os.path.join(filepath, 'Office'),
    #                os.path.join(filepath, 'Office_new')]

    # Case when we want to plot results for a specific scenario
    experiments = [os.path.join(filepath, 'Home')]

    # Iterate over scenarios
    for exp in experiments:
        # DTW distance bar plot
        generate_distance_plot_from_existing_files(exp)

        # Entropy bar plot
        generate_entropy_from_existing_files(exp)

    # Generate example figure of CO2 change when being affected by a humidifier
    filepath = 'C:/Users/mfomichev/Desktop/hardzipa-results/Light_Gas trends/FreshCalib'
    generate_example_change(filepath)
