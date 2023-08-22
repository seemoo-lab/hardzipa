import os
import plot_metrics
import json
import re
from glob import glob
from matplotlib import pyplot as plt
import matplotlib
import date_time_format
import numpy as np
import sys


def generate_distance_plot_from_existing_files(file_path):
    # Read result files
    files = glob(os.path.join(file_path, '**', 'distance_metrics_light_*'), recursive=True)

    # Iterate over result files
    for file in files:
        # Get file name
        file_name = os.path.basename(file)

        # Open a json file for reading
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

            # Get chunk's interval
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
            exclude_list = ['Light_4bulbs']

            # Path to save plots
            save_path = os.path.join(file_path, 'Light', 'distance', 'Plot_distance')

            # Create directory for saving plots
            os.makedirs(save_path, exist_ok=True)

            # Do some plotting
            plot_metrics.draw_experiment_distance_bar_plot(data, save_path=save_path, caption=caption, normalize=True,
                                                           exclude_list=exclude_list, arrange_order=is_not_followup,
                                                           is_light_followup=not is_not_followup)


def generate_entropy_from_existing_files(entropy_path):
    # Read result files
    files = glob(os.path.join(entropy_path, '**', 'light_entropy.json'), recursive=True)

    # Iterate over result files
    for file in files:

        # Open json file for reading
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

            # List for excluding some experiments, modality names, and bin sizes to evaluate entropy
            exclude_list = ['Light_4bulbs']
            modality = ['RGB', 'RGB raw', 'lum', 'lum raw']
            bins = ['20', '100']

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
                                                                  is_light_followup=not is_not_followup)


def glob_files(file_path, pattern):
    return glob(os.path.join(os.path.join(file_path, '**'), pattern), recursive=True)


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
                files_data[distance]['path'] = os.path.join(dir_path, 'results')
            else:
                files_data[distance]['path'] = os.path.join(dir_path, f'results{min_split}light')

        # Open a file
        with open(file, 'r') as f:
            while True:
                tmp = f.readline()
                if tmp == '':
                    break

                # Get timestamp and data
                tmp = tmp.replace(',', '')
                ts, d1, d2, d3 = tmp.split(' ')
                timestamp.append(date_time_format.get_date(ts))

                ld1.append(int(float(d1)))
                ld2.append(int(float(d2)))
                ld3.append(int(float(d3)))

        files_data[distance][device] = (timestamp, ld1, ld2, ld3)

    return files_data


def plot_distance_trend(file_path):
    # Set this up to avoid errors in submission systems such as HotCRP (i.e., proper font embeddings)
    matplotlib.rcParams.update({'pdf.fonttype': 42})
    matplotlib.rcParams.update({'ps.fonttype': 42})

    # Store RGB values
    red = []
    green = []
    blue = []

    # Store labels
    labels = []

    # Read source txt file
    files = glob_files(file_path, '*.txt')
    results = file_to_dict(files)

    # Distances
    distances = ['20cm', '40cm', '60cm', '80cm', '1m', '1m20cm', '1m40cm']

    # Iterate over distances
    for distance in distances:
        if distance == '1m':
            # We need this to set the distance to 100cm
            labels.append('100')
        else:
            labels.append(re.sub('[^0-9]', '', distance))

        # Iterate over colors
        for color in results[distance]:
            if color == 'path':
                continue
            if color == 'blue':
                blue.extend(results[distance][color][2][:50])
            if color == 'red':
                red.extend(results[distance][color][2][:50])
            if color == 'green':
                green.extend(results[distance][color][2][:50])

    # Values for X and Y axes labels
    x = np.arange(25, (len(labels) * 50) + 1, 50)
    y = np.arange(0, len(blue) + 1)

    # Set figure size
    plt.figure(figsize=(6.5, 4))

    # Adjust font for 10^x
    plt.rc('font', size=18)

    # Plot stuff
    plt.plot(red, color='red', label='Red', linewidth=6, linestyle=plot_metrics.get_line_style(0), zorder=3)
    plt.plot(green, color='green', label='Green', linewidth=6, linestyle=plot_metrics.get_line_style(1), zorder=3)
    plt.plot(blue, color='blue', label='Blue', linewidth=6, linestyle=plot_metrics.get_line_style(2), zorder=3)

    # Add 10^x for ticks (Y-axis)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useMathText=True)

    # Grid
    plt.grid(True, axis='y', zorder=0)

    # Axes names and font sizes + ticks font size
    plt.xlabel('Distance (meters)', fontsize=26)
    plt.ylabel('RGB Value', fontsize=26)
    plt.tick_params(axis='both', labelsize=22)
    plt.xticks(list(x)[::2], labels=[float(label) / 100 for label in labels][::2])

    # Add yticks
    plt.yticks(np.arange(0, 6.1, step=2) * 10000)

    # Have ticks on left and right side of Y-axis
    plt.gca().yaxis.set_ticks_position('both')

    # Ticks inside the plot look nicer IMHO
    plt.tick_params(axis='y', direction='in')

    # Set up a legend
    lgnd = plt.legend(loc='best', ncol=3, borderaxespad=0.,
                      fontsize=21, handlelength=1.2, handletextpad=0.01, columnspacing=0.7)

    # Increase line thickness of legend markers
    for line in lgnd.get_lines():
        line.set_linewidth(7)

    # Save plots (let's do both PDF and SVG)
    to_save = os.path.join(file_path, 'rgb-light')
    plt.savefig(to_save + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    plt.savefig(to_save + '.svg', format='svg', dpi=1000, bbox_inches='tight')
    plt.close('all')


def get_entropy_example_from_real_data(file_path):
    # Get data files from directory
    files = glob(os.path.join(os.path.join(file_path, '**'), 'RGB.txt'), recursive=True)
    files.sort()

    # Load data
    results = file_to_dict(files)

    # Get samples with high and low entropy
    high_entropy = plot_metrics.normalize_mean_sub(results['Office_entropy']['101'][1][1000:1350])
    low_entropy = plot_metrics.normalize_mean_sub(results['Office_base']['101'][1][1000:1350])

    # Get number of bins for the RGB modality
    r_max = max(np.max(high_entropy), np.max(low_entropy))
    r_min = min(np.min(high_entropy), np.min(low_entropy))

    bins = np.histogram_bin_edges((), bins=15, range=(r_min - 1, r_max + 1))

    # Do the plotting
    plot_metrics.example_entropy(high_entropy, low_entropy, bins, file_path)


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

    # Generate example figure of RGB readings being affected by the color light
    filepath = 'C:/Users/mfomichev/Desktop/hardzipa-results/Light_Gas trends/Distance/Indirect'
    plot_distance_trend(filepath)

    # Plot low/high entropy example figures
    filepath = 'C:/Users/mfomichev/Desktop/hardzipa-data/Office'
    get_entropy_example_from_real_data(filepath)
